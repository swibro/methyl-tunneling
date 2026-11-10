"""Shared code for benchmarking and fine-tuning MACE-OFF force fields on CP2K labels.

The per-model scripts in ``models/mace-off*/`` are thin wrappers around the three
entry points at the bottom of this file. They mirror the UMA entry points in
``common.py`` and reuse its frame loading, split, metrics and reports, so the output
has the same format.

  run_zero_shot       benchmark the pretrained model on ``work/labeled.extxyz``
  run_finetune        split the labels, write a mace_run_train config, train, keep
                      the best-validation model as ``tuned.pt``, benchmark it
  run_benchmark_tuned benchmark an existing fine-tuned model

MACE needs ``e3nn==0.4.4`` and fairchem needs ``e3nn>=0.5``, so these scripts run
from a separate venv (``menv-mace``), not from ``menv``.

MACE-OFF labels are wB97M-D3(BJ)/def2-TZVPPD, not the PBE-D3(BJ) of the CP2K labels,
so a zero-shot difference from CP2K mixes functional error with model error. Those
labels include dispersion, but the model only sees it within its receptive field
(2 x r_max), so no D3 correction is added here. Fine-tuning moves the model towards
PBE-D3(BJ), so a tuned model is no longer a wB97M-D3(BJ) comparison.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from ase.io import write

import common

WEIGHTS_URL = "https://raw.githubusercontent.com/ACEsuit/mace-off/main"
MODEL_URLS = {
  "mace-off23-s": f"{WEIGHTS_URL}/mace_off23/MACE-OFF23_small.model",
  "mace-off23-m": f"{WEIGHTS_URL}/mace_off23/MACE-OFF23_medium.model",
  "mace-off23-l": f"{WEIGHTS_URL}/mace_off23/MACE-OFF23_large.model",
  "mace-off24-m": f"{WEIGHTS_URL}/mace_off24/MACE-OFF24_medium.model",
}
LEVEL_OF_THEORY = "wB97M-D3(BJ)/def2-TZVPPD"
# Total energies are ~4e5 eV (all-electron references), where float32 spacing is 31 meV.
DTYPE = "float64"
TRAIN_NAME = "finetune"


# ---------------------------------------------------------------- models

def make_calc(name_or_path: str, device: str):
  """Build a MACE ASE calculator for a pretrained or fine-tuned model.

  Parameters
  ----------
  name_or_path : str
    Key of ``MODEL_URLS`` (weights are downloaded to ``~/.cache/mace`` on first
    use) or a path to a fine-tuned model.
  device : {"cpu", "cuda"}
    Torch device.

  Returns
  -------
  MACECalculator
    Calculator predicting energy, forces and stress.
  """
  try:
    from mace.calculators import mace_off
  except ImportError as err:
    raise SystemExit(f"Cannot import mace ({err}); run this from menv-mace, not menv.") from err
  model = MODEL_URLS.get(str(name_or_path), str(name_or_path))
  return mace_off(model=model, device=device, default_dtype=DTYPE)


def check_elements(frames: list, calc, model_name: str) -> None:
  """Stop with a clear message if a frame has an element the model does not cover.

  Parameters
  ----------
  frames : list of Atoms
    Frames to evaluate.
  calc : MACECalculator
    Calculator built by ``make_calc``.
  model_name : str
    Name used in the message.

  Returns
  -------
  None
  """
  supported = {int(z) for z in calc.z_table.zs}
  symbols = {s for f in frames for s, z in zip(f.get_chemical_symbols(), f.numbers) if int(z) not in supported}
  if symbols:
    raise SystemExit(f"{model_name} does not cover {sorted(symbols)}; it has Z = {sorted(supported)}")


def evaluate(frames: list, calc, model_name: str) -> tuple:
  """Run a model on frames and compare it with the CP2K labels.

  Parameters
  ----------
  frames : list of Atoms
    Labeled frames.
  calc : MACECalculator
    Calculator built by ``make_calc``.
  model_name : str
    Name used in error messages.

  Returns
  -------
  tuple of (list of dict, dict)
    Output of ``common.predict`` and of ``common.benchmark``.
  """
  check_elements(frames, calc, model_name)
  preds = common.predict(frames, calc)
  return preds, common.benchmark(frames, preds)


# ---------------------------------------------------------------- fine-tuning

def weights_file(model_name: str) -> Path:
  """Download the pretrained weights if needed and return the cached file.

  Parameters
  ----------
  model_name : str
    Key of ``MODEL_URLS``.

  Returns
  -------
  Path
    File in the MACE cache directory.
  """
  make_calc(model_name, "cpu")
  from mace.tools.utils import get_cache_dir

  return Path(get_cache_dir()) / MODEL_URLS[model_name].rsplit("/", 1)[1]


def write_mace_xyz(frames: list, path: Path) -> None:
  """Write labeled frames with the REF_* keys that mace_run_train reads.

  MACE warns that the plain "energy", "forces" and "stress" keys are unsafe with
  ASE >= 3.23, so the labels are moved out of the calculator.

  Parameters
  ----------
  frames : list of Atoms
    Labeled frames.
  path : Path
    Output extended-XYZ file.

  Returns
  -------
  None
  """
  out = []
  for frame in frames:
    atoms = frame.copy()
    atoms.info["REF_energy"] = frame.get_potential_energy()
    atoms.arrays["REF_forces"] = frame.get_forces()
    # REF_forces: (n_atoms, 3)
    if "stress" in frame.calc.results:
      atoms.info["REF_stress"] = frame.get_stress()
      # REF_stress: (6,), Voigt, eV/A^3
    out.append(atoms)
  write(path, out)


def build_config(ft_dir: Path, args: argparse.Namespace, weights: Path) -> Path:
  """Write the mace_run_train config for fine-tuning.

  ``E0s: estimated`` refits the atomic reference energies so the model matches the
  CP2K energy scale. With a single composition this is a min-norm solution: correct
  for that composition only, as for the UMA element references.

  Parameters
  ----------
  ft_dir : Path
    Fine-tune working directory (contains ``mace_data/``).
  args : argparse.Namespace
    Parsed fine-tune arguments.
  weights : Path
    Pretrained model file.

  Returns
  -------
  Path
    The config file to pass to ``mace_run_train --config``.
  """
  cfg = {
    "name": TRAIN_NAME,
    "foundation_model": str(weights),
    "multiheads_finetuning": False,
    "train_file": str(ft_dir / "mace_data" / "train.extxyz"),
    "valid_file": str(ft_dir / "mace_data" / "val.extxyz"),
    "energy_key": "REF_energy",
    "forces_key": "REF_forces",
    "stress_key": "REF_stress",
    "E0s": "estimated",
    "loss": "stress" if args.regression == "efs" else "weighted",
    "energy_weight": args.energy_weight,
    "forces_weight": args.forces_weight,
    "stress_weight": args.stress_weight,
    "lr": args.lr,
    "batch_size": args.batch_size,
    "valid_batch_size": args.batch_size,
    "max_num_epochs": args.epochs,
    "eval_interval": args.eval_every,
    "ema": True,
    "ema_decay": 0.99,
    "amsgrad": True,
    "default_dtype": DTYPE,
    "device": args.device,
    "seed": args.seed,
    "work_dir": str(ft_dir / "runs"),
    "save_cpu": True,
    "plot": False,
  }
  path = ft_dir / "finetune.yaml"
  with open(path, "w") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False)
  return path


def train_command(config: Path) -> list:
  """Command that launches mace_run_train from the same venv as this interpreter.

  Parameters
  ----------
  config : Path
    Config file written by ``build_config``.

  Returns
  -------
  list of str
    Command and arguments.
  """
  return [str(Path(sys.executable).parent / "mace_run_train"), "--config", str(config)]


def collect_model(ft_dir: Path, keep_checkpoints: bool) -> Path:
  """Move the trained model to ``<ft_dir>/tuned.pt`` and prune the training output.

  Parameters
  ----------
  ft_dir : Path
    Fine-tune working directory.
  keep_checkpoints : bool
    If True, leave the training checkpoints in place and copy instead of move.

  Returns
  -------
  Path
    Path of the tuned model.
  """
  runs = ft_dir / "runs"
  final = runs / f"{TRAIN_NAME}.model"
  if not final.exists():
    raise SystemExit(f"Training finished without {final}; check {runs / 'logs'}")
  tuned = ft_dir / common.TUNED_CHECKPOINT
  if keep_checkpoints:
    shutil.copy2(final, tuned)
  else:
    shutil.move(str(final), tuned)
    shutil.rmtree(runs / "checkpoints", ignore_errors=True)
  print(f"Tuned model: {tuned} ({tuned.stat().st_size / 1e6:.1f} MB)")
  return tuned


# ---------------------------------------------------------------- entry points

def run_zero_shot(model_name: str, model_dir: Path, default_device: str = common.DEFAULT_DEVICE, argv: list = None) -> None:
  """CLI: benchmark the pretrained model on every labeled frame.

  Parameters
  ----------
  model_name : str
    Key of ``MODEL_URLS``.
  model_dir : Path
    ``models/<model_name>``; results go to ``<model_dir>/zero_shot``.
  default_device : str, optional
    Device used when ``--device`` is not given.
  argv : list of str, optional
    Arguments (defaults to ``sys.argv[1:]``).

  Returns
  -------
  None
  """
  parser = common._base_parser(f"Zero-shot benchmark of pretrained {model_name} on CP2K labels.", default_device)
  parser.add_argument("--out", type=Path, default=model_dir / "zero_shot")
  args = parser.parse_args(argv)
  frames = common.load_labeled(args.labeled)
  print(f"{len(frames)} labeled frames from {args.labeled}; loading {model_name} on {args.device}")
  preds, results = evaluate(frames, make_calc(model_name, args.device), model_name)
  meta = {"model": model_name, "tuned": False, "task": None, "level_of_theory": LEVEL_OF_THEORY, "dtype": DTYPE,
          "device": args.device, "labeled": str(args.labeled)}
  common.write_report(args.out, frames, preds, results, meta)


def run_finetune(
  model_name: str,
  model_dir: Path,
  default_device: str = common.DEFAULT_DEVICE,
  cpu_ok: bool = True,
  argv: list = None,
) -> None:
  """CLI: fine-tune the pretrained model on the labeled frames, then benchmark it.

  Parameters
  ----------
  model_name : str
    Key of ``MODEL_URLS``.
  model_dir : Path
    ``models/<model_name>``; work goes to ``<model_dir>/finetune``.
  default_device : str, optional
    Device used when ``--device`` is not given.
  cpu_ok : bool, optional
    False for models too slow to train on CPU; then ``--force`` is required.
  argv : list of str, optional
    Arguments (defaults to ``sys.argv[1:]``).

  Returns
  -------
  None
  """
  parser = common._base_parser(f"Fine-tune {model_name} on CP2K labels and benchmark it.", default_device)
  parser.add_argument("--out", type=Path, default=model_dir / "finetune")
  parser.add_argument("--split", choices=["by-type", "random"], default="by-type",
                      help="by-type: rigid_scan = test, part of rattles = val (default)")
  parser.add_argument("--val-frac", type=float, default=0.2)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--regression", choices=["efs", "ef"], default="efs", help="train on energy+forces(+stress)")
  parser.add_argument("--epochs", type=int, default=100)
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--energy-weight", type=float, default=100.0, help="per-atom energy loss weight")
  parser.add_argument("--forces-weight", type=float, default=10.0)
  parser.add_argument("--stress-weight", type=float, default=100.0)
  parser.add_argument("--eval-every", type=int, default=1, help="epochs between validations")
  parser.add_argument("--allow-tiny", action="store_true", help="smoke test: reuse train frames as val if a split is empty")
  parser.add_argument("--keep-checkpoints", action="store_true", help="keep the training checkpoints")
  parser.add_argument("--prepare-only", action="store_true", help="write data and config, do not train")
  parser.add_argument("--force", action="store_true", help="train even where it is not recommended (large model on CPU)")
  args = parser.parse_args(argv)

  if args.device == "cpu" and not cpu_ok and not args.force and not args.prepare_only:
    raise SystemExit(f"{model_name} is slow to fine-tune on CPU. Use --device cuda or --force.")
  ft_dir = args.out.resolve()
  if (ft_dir / "runs").exists():
    raise SystemExit(f"{ft_dir / 'runs'} exists; move it away or pass a new --out")
  common.mark_untracked(ft_dir)

  frames = common.load_labeled(args.labeled)
  splits = common.split_frames(frames, args.split, args.val_frac, args.seed)
  if not splits["val"] or not splits["train"]:
    if not args.allow_tiny:
      sizes = {k: len(v) for k, v in splits.items()}
      raise SystemExit(f"Empty split {sizes}; label more frames or use --allow-tiny")
    pool = splits["train"] or splits["val"] or splits["test"]
    splits["train"] = splits["train"] or pool
    splits["val"] = splits["val"] or pool
    print("WARNING: --allow-tiny reused frames across splits; metrics are not meaningful.")
  print("Split: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
  (ft_dir / "split").mkdir(parents=True, exist_ok=True)
  (ft_dir / "mace_data").mkdir(parents=True, exist_ok=True)
  for name, part in splits.items():
    if part:
      write(ft_dir / "split" / f"{name}.extxyz", part)
  for name in ("train", "val"):
    write_mace_xyz(splits[name], ft_dir / "mace_data" / f"{name}.extxyz")

  config = build_config(ft_dir, args, weights_file(model_name))
  print(f"Wrote {config}")
  if args.prepare_only:
    print("Train with: " + " ".join(train_command(config)))
    return

  subprocess.run(train_command(config), check=True)
  tuned = collect_model(ft_dir, args.keep_checkpoints)
  calc = make_calc(tuned, args.device)
  meta = {"model": f"{model_name} fine-tuned", "tuned": True, "base_model": model_name, "task": None, "dtype": DTYPE,
          "device": args.device, "labeled": str(args.labeled), "split": args.split, "epochs": args.epochs,
          "lr": args.lr, "regression": args.regression, "energy_weight": args.energy_weight,
          "forces_weight": args.forces_weight, "stress_weight": args.stress_weight}
  for name, part in splits.items():
    if part:
      print(f"Benchmark on {name} split:")
      preds, results = evaluate(part, calc, model_name)
      common.write_report(ft_dir / "benchmark" / name, part, preds, results, {**meta, "eval_split": name})


def run_benchmark_tuned(model_name: str, model_dir: Path, default_device: str = common.DEFAULT_DEVICE, argv: list = None) -> None:
  """CLI: benchmark an existing fine-tuned model on a labeled file.

  Parameters
  ----------
  model_name : str
    Key of ``MODEL_URLS`` (for labels only).
  model_dir : Path
    ``models/<model_name>``; the model defaults to ``<model_dir>/finetune/tuned.pt``.
  default_device : str, optional
    Device used when ``--device`` is not given.
  argv : list of str, optional
    Arguments (defaults to ``sys.argv[1:]``).

  Returns
  -------
  None
  """
  parser = common._base_parser(f"Benchmark a fine-tuned {model_name} model on CP2K labels.", default_device)
  parser.add_argument("--checkpoint", type=Path, default=model_dir / "finetune" / common.TUNED_CHECKPOINT)
  parser.add_argument("--out", type=Path, default=model_dir / "finetune" / "benchmark" / "custom")
  args = parser.parse_args(argv)
  if not args.checkpoint.exists():
    raise SystemExit(f"{args.checkpoint} not found; run finetune.py first")
  frames = common.load_labeled(args.labeled)
  preds, results = evaluate(frames, make_calc(args.checkpoint, args.device), model_name)
  meta = {"model": f"{model_name} fine-tuned", "tuned": True, "checkpoint": str(args.checkpoint), "dtype": DTYPE,
          "device": args.device, "labeled": str(args.labeled)}
  common.write_report(args.out, frames, preds, results, meta)
  zero = model_dir / "zero_shot" / "results.json"
  if zero.exists():
    with open(zero) as fh:
      base = json.load(fh)["metrics"]["all"]
    print(f"zero-shot (all frames): F MAE {base['force_mae_eV_A']:.4f} eV/A, "
          f"dE MAE {base['rel_energy_mae_meV_per_atom']:.4f} meV/atom")
