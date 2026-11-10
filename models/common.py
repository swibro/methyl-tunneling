"""Shared code for benchmarking and fine-tuning fairchem UMA models (task "omc") on CP2K labels.

The per-model scripts in ``models/<model-name>/`` are thin wrappers around the
three entry points at the bottom of this file:

  run_zero_shot       benchmark the pretrained model on ``work/labeled.extxyz``
  run_finetune        split the labels, build ASE-LMDBs + a fairchem YAML, train,
                      keep only the final inference checkpoint, benchmark it
  run_benchmark_tuned benchmark an existing fine-tuned checkpoint

Absolute energies are not comparable between CP2K (GTH pseudopotentials) and
OMC25 (VASP PAW), so zero-shot energies are compared relative to the
``reference`` frame (or the frame mean if there is none).
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from ase import Atoms
from ase.io import read, write

METHYL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LABELED = METHYL_DIR / "work" / "labeled.extxyz"
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
TASK_NAME = "omc"
# Change this to "cuda" to make every script default to the GPU.
DEFAULT_DEVICE = "cpu"
REGRESSION_TEMPLATES = {
  "ef": "uma_conserving_data_task_energy_force",
  "efs": "uma_conserving_data_task_energy_force_stress",
}
FINETUNE_TEMPLATE = "uma_sm_finetune_template.yaml"
TUNED_CHECKPOINT = "tuned.pt"
RUN_TIMESTAMP = "run"


def mark_untracked(path: Path) -> None:
  """Create ``path`` and put a ``.gitignore`` containing ``*`` in it, so git skips the whole directory.

  Parameters
  ----------
  path : Path
    Generated output directory.

  Returns
  -------
  None
  """
  path.mkdir(parents=True, exist_ok=True)
  marker = path / ".gitignore"
  if not marker.exists():
    marker.write_text("# generated output; ignored by git\n*\n")


# ---------------------------------------------------------------- data

def load_labeled(path: Path) -> list:
  """Read labeled frames, check they carry DFT results, drop duplicate frame IDs.

  Parameters
  ----------
  path : Path
    Labeled extended-XYZ file written by ``make_dataset.py label``.

  Returns
  -------
  list of Atoms
    Frames in file order, each with a SinglePointCalculator holding energy,
    forces and (usually) stress.
  """
  frames, seen = [], set()
  for atoms in read(path, index=":"):
    if atoms.calc is None or "energy" not in atoms.calc.results or "forces" not in atoms.calc.results:
      raise ValueError(f"frame {atoms.info.get('frame_id')} in {path} has no energy/forces")
    fid = int(atoms.info.get("frame_id", len(frames)))
    if fid in seen:
      continue
    seen.add(fid)
    frames.append(atoms)
  if not frames:
    raise SystemExit(f"No labeled frames in {path}")
  return frames


def split_frames(frames: list, mode: str = "by-type", val_frac: float = 0.2, seed: int = 0) -> dict:
  """Split labeled frames into train / val / test.

  ``by-type`` holds out every ``rigid_scan`` frame as test (the rotational
  potential is the quantity of interest, so the model never sees it), takes a
  seeded ``val_frac`` of the ``rattle`` frames as validation, and trains on the
  rest. ``random`` puts a seeded ``val_frac`` of all frames in val and leaves
  test empty.

  Parameters
  ----------
  frames : list of Atoms
    Labeled frames with ``info["config_type"]``.
  mode : {"by-type", "random"}, optional
    Split strategy. Defaults to "by-type".
  val_frac : float, optional
    Fraction for validation. Defaults to 0.2.
  seed : int, optional
    Seed for the shuffle. Defaults to 0.

  Returns
  -------
  dict
    Keys "train", "val", "test", each a list of Atoms.
  """
  rng = random.Random(seed)
  if mode == "random":
    shuffled = list(frames)
    rng.shuffle(shuffled)
    n_val = max(1, round(val_frac * len(shuffled)))
    return {"train": shuffled[n_val:], "val": shuffled[:n_val], "test": []}
  if mode != "by-type":
    raise ValueError(f"unknown split mode {mode!r}")
  test = [f for f in frames if f.info.get("config_type") == "rigid_scan"]
  rattles = [f for f in frames if f.info.get("config_type") == "rattle"]
  rng.shuffle(rattles)
  n_val = max(1, round(val_frac * len(rattles))) if rattles else 0
  val_ids = {int(f.info["frame_id"]) for f in rattles[:n_val]}
  val = [f for f in frames if int(f.info["frame_id"]) in val_ids]
  train = [f for f in frames if f.info.get("config_type") != "rigid_scan" and int(f.info["frame_id"]) not in val_ids]
  return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------- models

def make_calc(name_or_path: str, device: str):
  """Build a FAIRChem ASE calculator on the omc task.

  Parameters
  ----------
  name_or_path : str
    Pretrained model name (e.g. "uma-s-1p1", downloaded from Hugging Face) or a
    path to a fine-tuned ``inference_ckpt.pt``.
  device : {"cpu", "cuda"}
    Torch device.

  Returns
  -------
  FAIRChemCalculator
    Calculator predicting energy, forces and (if the head supports it) stress.
  """
  from fairchem.core import FAIRChemCalculator

  return FAIRChemCalculator.from_model_checkpoint(str(name_or_path), task_name=TASK_NAME, device=device)


def predict(frames: list, calc) -> list:
  """Run the MLIP on every frame.

  Parameters
  ----------
  frames : list of Atoms
    Labeled frames (their DFT results are left untouched).
  calc : Calculator
    MLIP calculator.

  Returns
  -------
  list of dict
    Per frame: "energy" (float), "forces" ((n_atoms, 3) array), "stress"
    ((6,) Voigt array, or None if the model does not predict stress).
  """
  from ase.calculators.calculator import PropertyNotImplementedError

  preds = []
  for k, frame in enumerate(frames):
    atoms = Atoms(frame)
    atoms.calc = calc
    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    # forces: (n_atoms, 3)
    try:
      stress = atoms.get_stress()
      # stress: (6,), Voigt, eV/A^3
    except PropertyNotImplementedError:
      stress = None
    preds.append({"energy": energy, "forces": forces, "stress": stress})
    print(f"  [{k + 1}/{len(frames)}] frame {frame.info.get('frame_id')} "
          f"{frame.info.get('config_type', '?'):12s} E_model={energy:.4f} eV")
  return preds


# ---------------------------------------------------------------- metrics

def fit_barrier(phi_deg: np.ndarray, energy: np.ndarray) -> dict:
  """Least-squares fit of V(phi) = V3/2 (1 - cos 3(phi - phi0)) + V6/2 (1 - cos 6(phi - phi0)) + c.

  Linearised as E = c + a3 cos3phi + b3 sin3phi + a6 cos6phi + b6 sin6phi; V3 and
  phi0 come from the threefold term; V6 is the signed sixfold amplitude about phi0.

  Parameters
  ----------
  phi_deg : np.ndarray
    Rotation angles in degrees. Shape: (n_angles,).
  energy : np.ndarray
    Energies in eV, any offset. Shape: (n_angles,).

  Returns
  -------
  dict
    "V3_meV", "V6_meV", "phi0_deg", "rmse_meV"; empty if fewer than 5 angles.
  """
  if len(phi_deg) < 5:
    return {}
  phi = np.radians(phi_deg)
  # phi: (n_angles,)
  design = np.column_stack([np.ones_like(phi), np.cos(3 * phi), np.sin(3 * phi), np.cos(6 * phi), np.sin(6 * phi)])
  # design: (n_angles, 5)
  coef, *_ = np.linalg.lstsq(design, energy, rcond=None)
  # coef: (5,)
  amp3 = np.hypot(coef[1], coef[2])
  # -(V3/2) cos 3(phi - phi0) = a3 cos3phi + b3 sin3phi  ->  phi0 = atan2(-b3, -a3) / 3
  phi0 = np.arctan2(-coef[2], -coef[1]) / 3.0
  # V6 is signed: the projection of the sixfold term onto cos 6(phi - phi0)
  v6 = -2.0 * (coef[3] * np.cos(6 * phi0) + coef[4] * np.sin(6 * phi0))
  resid = energy - design @ coef
  # resid: (n_angles,)
  return {
    "V3_meV": float(2 * amp3 * 1e3),
    "V6_meV": float(v6 * 1e3),
    "phi0_deg": float(np.degrees(phi0)),
    "rmse_meV": float(np.sqrt(np.mean(resid ** 2)) * 1e3),
  }


def _reference_energy(frames: list, energies: np.ndarray) -> float:
  """Energy of the ``reference`` frame, or the mean if there is none."""
  for frame, e in zip(frames, energies):
    if frame.info.get("config_type") == "reference":
      return float(e)
  return float(np.mean(energies))


def _group_metrics(frames: list, preds: list, e_dft_rel: np.ndarray, e_mlp_rel: np.ndarray) -> dict:
  """Energy/force/stress errors for one group of frames."""
  n_atoms = np.array([len(f) for f in frames], dtype=float)
  # n_atoms: (n_frames,)
  de = (e_mlp_rel - e_dft_rel) * 1e3
  # de: (n_frames,), meV per cell
  f_dft = np.concatenate([f.get_forces() for f in frames])
  f_mlp = np.concatenate([p["forces"] for p in preds])
  # f_dft, f_mlp: (sum n_atoms, 3)
  is_h = np.concatenate([np.array(f.get_chemical_symbols()) == "H" for f in frames])
  # is_h: (sum n_atoms,)
  cos = np.sum(f_dft * f_mlp, axis=1) / (np.linalg.norm(f_dft, axis=1) * np.linalg.norm(f_mlp, axis=1) + 1e-12)
  # cos: (sum n_atoms,)
  out = {
    "n_frames": len(frames),
    "rel_energy_mae_meV": float(np.mean(np.abs(de))),
    "rel_energy_mae_meV_per_atom": float(np.mean(np.abs(de) / n_atoms)),
    "force_mae_eV_A": float(np.mean(np.abs(f_mlp - f_dft))),
    "force_rmse_eV_A": float(np.sqrt(np.mean((f_mlp - f_dft) ** 2))),
    "force_mae_H_eV_A": float(np.mean(np.abs(f_mlp[is_h] - f_dft[is_h]))),
    "force_cosine_mean": float(np.mean(cos)),
  }
  pairs = [(f.calc.results.get("stress"), p["stress"]) for f, p in zip(frames, preds)]
  pairs = [(np.asarray(s_dft), s_mlp) for s_dft, s_mlp in pairs if s_dft is not None and s_mlp is not None]
  if pairs:
    s_err = np.array([s_mlp - s_dft for s_dft, s_mlp in pairs]) * 160.21766
    # s_err: (n_with_stress, 6), GPa
    out["stress_mae_GPa"] = float(np.mean(np.abs(s_err)))
  return out


def benchmark(frames: list, preds: list) -> dict:
  """Compare MLIP predictions with the CP2K labels.

  Parameters
  ----------
  frames : list of Atoms
    Labeled frames.
  preds : list of dict
    Output of ``predict`` for the same frames.

  Returns
  -------
  dict
    "all" and one entry per config_type with error metrics, plus "scan" (DFT and
    model V3/V6/phi0 fits) when rigid-scan frames are present, and "absolute_offset_eV"
    (mean model minus DFT total energy, informative only for fine-tuned models).
  """
  e_dft = np.array([f.get_potential_energy() for f in frames])
  e_mlp = np.array([p["energy"] for p in preds])
  # e_dft, e_mlp: (n_frames,)
  e_dft_rel = e_dft - _reference_energy(frames, e_dft)
  e_mlp_rel = e_mlp - _reference_energy(frames, e_mlp)
  # e_dft_rel, e_mlp_rel: (n_frames,)
  results = {
    "absolute_offset_eV": float(np.mean(e_mlp - e_dft)),
    "all": _group_metrics(frames, preds, e_dft_rel, e_mlp_rel),
  }
  types = sorted({f.info.get("config_type", "unknown") for f in frames})
  for ctype in types:
    idx = [k for k, f in enumerate(frames) if f.info.get("config_type", "unknown") == ctype]
    results[ctype] = _group_metrics([frames[k] for k in idx], [preds[k] for k in idx], e_dft_rel[idx], e_mlp_rel[idx])

  scan = [k for k, f in enumerate(frames) if f.info.get("config_type") == "rigid_scan"]
  # labeled.extxyz is in labeling order (relabeled frames are appended), so sort by angle
  # so the scan lists in results.json and the line in scan.png follow phi.
  scan.sort(key=lambda k: frames[k].info["phi_deg"])
  if scan:
    phi = np.array([frames[k].info["phi_deg"] for k in scan])
    # phi: (n_scan,)
    results["scan"] = {
      "phi_deg": phi.tolist(),
      "dE_dft_meV": ((e_dft[scan] - e_dft[scan].min()) * 1e3).tolist(),
      "dE_model_meV": ((e_mlp[scan] - e_mlp[scan].min()) * 1e3).tolist(),
      "fit_dft": fit_barrier(phi, e_dft[scan]),
      "fit_model": fit_barrier(phi, e_mlp[scan]),
    }
  return results


# ---------------------------------------------------------------- reporting

def draw_scan(ax, scan: dict, label: str):
  """Draw DFT and model rigid-scan energy profiles.

  Parameters
  ----------
  ax : Axes
    Target axes; labels and legend are set here.
  scan : dict
    The "scan" entry of ``benchmark`` output.
  label : str
    Legend label for the model curve.

  Returns
  -------
  Line2D
    The model line.
  """
  ax.plot(scan["phi_deg"], scan["dE_dft_meV"], marker="o", color="0.2", label="CP2K PBE-D3(BJ)")
  (line,) = ax.plot(scan["phi_deg"], scan["dE_model_meV"], marker="s", linestyle="--", label=label)
  ax.set(xlabel="methyl rotation φ (deg)", ylabel="ΔE (meV / cell)", title="Rigid rotor scan")
  ax.legend(frameon=False)
  return line


def write_report(outdir: Path, frames: list, preds: list, results: dict, meta: dict) -> None:
  """Write results.json, predictions.extxyz and (if available) scan.png.

  Parameters
  ----------
  outdir : Path
    Output directory, created if missing.
  frames : list of Atoms
    Labeled frames.
  preds : list of dict
    Model predictions for the same frames.
  results : dict
    Output of ``benchmark``.
  meta : dict
    Run description (model, device, split, ...) stored alongside the metrics.

  Returns
  -------
  None
  """
  mark_untracked(outdir)
  with open(outdir / "results.json", "w") as fh:
    json.dump({"meta": meta, "metrics": results}, fh, indent=2)

  out_frames = []
  for frame, pred in zip(frames, preds):
    atoms = frame.copy()
    atoms.calc = frame.calc
    atoms.info["model_energy"] = pred["energy"]
    if pred["stress"] is not None:
      atoms.info["model_stress"] = np.asarray(pred["stress"])
    atoms.arrays["model_forces"] = np.asarray(pred["forces"])
    out_frames.append(atoms)
  write(outdir / "predictions.extxyz", out_frames)

  if "scan" in results:
    from matplotlib.figure import Figure

    fig = Figure(figsize=(5.0, 3.5), layout="constrained")
    draw_scan(fig.subplots(), results["scan"], meta.get("model", "model"))
    fig.savefig(outdir / "scan.png", dpi=200)
  print_summary(results)
  print(f"Wrote {outdir}/results.json")


def print_summary(results: dict) -> None:
  """Print a one-line-per-group table of the benchmark metrics."""
  print(f"{'group':14s} {'n':>3s} {'dE MAE meV/at':>14s} {'F MAE eV/A':>11s} {'F MAE H':>8s} {'cos':>6s} {'S MAE GPa':>10s}")
  for group, m in results.items():
    if not isinstance(m, dict) or "n_frames" not in m:
      continue
    stress = f"{m['stress_mae_GPa']:10.4f}" if "stress_mae_GPa" in m else f"{'-':>10s}"
    print(f"{group:14s} {m['n_frames']:3d} {m['rel_energy_mae_meV_per_atom']:14.4f} "
          f"{m['force_mae_eV_A']:11.4f} {m['force_mae_H_eV_A']:8.4f} {m['force_cosine_mean']:6.3f} {stress}")
  if "scan" in results:
    for who in ("dft", "model"):
      fit = results["scan"][f"fit_{who}"]
      if fit:
        print(f"  {who:5s} V3={fit['V3_meV']:.2f} meV V6={fit['V6_meV']:.2f} meV phi0={fit['phi0_deg']:.2f} deg")


# ---------------------------------------------------------------- fine-tuning

def write_aselmdb(frames: list, out_dir: Path) -> None:
  """Write frames to one ASE-LMDB file plus the metadata.npz fairchem's sampler reads.

  Same format as ``fairchem.core.scripts.create_finetune_dataset.launch_processing``,
  but in-process (no multiprocessing pool, which is unnecessary for tens of frames).

  Parameters
  ----------
  frames : list of Atoms
    Labeled frames.
  out_dir : Path
    Directory for ``data.0000.aselmdb`` and ``metadata.npz``.

  Returns
  -------
  None
  """
  from fairchem.core.scripts.create_finetune_dataset import write_ase_db

  out_dir.mkdir(parents=True, exist_ok=True)
  src = out_dir / "frames.extxyz"
  write(src, frames)
  _, natoms, _, failed = write_ase_db((out_dir / "data.0000.aselmdb", [str(src)], 0))
  src.unlink()
  if failed:
    raise RuntimeError(f"could not write frames to LMDB: {failed[:3]}")
  np.savez_compressed(out_dir / "metadata.npz", natoms=np.array(natoms))


def normalizers(frames: list) -> tuple:
  """Force RMS and per-element linear energy references from the training frames.

  Mirrors ``compute_normalizer_and_linear_reference`` from fairchem. With a single
  composition the lstsq element references are a min-norm solution: correct
  for that composition only.

  Parameters
  ----------
  frames : list of Atoms
    Training frames.

  Returns
  -------
  tuple of (float, list of float)
    Force RMS (eV/A) and 100 element reference energies (eV) indexed by Z.
  """
  from fairchem.core.scripts.create_finetune_dataset import compute_lin_ref

  forces = np.concatenate([f.get_forces() for f in frames])
  # forces: (sum n_atoms, 3)
  force_rms = float(np.sqrt(np.mean(forces ** 2)))
  coeff = compute_lin_ref([f.get_atomic_numbers() for f in frames], [f.get_potential_energy() for f in frames])
  return force_rms, [float(c) for c in coeff]


def build_config(ft_dir: Path, args: argparse.Namespace, model_name: str, force_rms: float, elem_refs: list) -> Path:
  """Write the fairchem Hydra config (main YAML + data/<task>.yaml) for fine-tuning.

  Starts from the templates in ``models/templates`` (copied from fairchem tag
  fairchem_core-2.23.0) and overrides device, logger, paths, schedule and
  checkpoint retention.

  Parameters
  ----------
  ft_dir : Path
    Fine-tune working directory (contains ``lmdb/train`` and ``lmdb/val``).
  args : argparse.Namespace
    Parsed fine-tune arguments.
  model_name : str
    Base UMA model name.
  force_rms : float
    Force normaliser, eV/A.
  elem_refs : list of float
    Element reference energies, length 100.

  Returns
  -------
  Path
    The main config file to pass to ``fairchem -c``.
  """
  data_name = REGRESSION_TEMPLATES[args.regression]
  with open(TEMPLATE_DIR / "data" / f"{data_name}.yaml") as fh:
    data = yaml.safe_load(fh)
  data["dataset_name"] = TASK_NAME
  data["normalizer_rmsd"] = force_rms
  data["elem_refs"] = elem_refs
  data["train_dataset"]["splits"]["train"]["src"] = str(ft_dir / "lmdb" / "train")
  data["val_dataset"]["splits"]["val"]["src"] = str(ft_dir / "lmdb" / "val")
  (ft_dir / "data").mkdir(parents=True, exist_ok=True)
  with open(ft_dir / "data" / f"{data_name}.yaml", "w") as fh:
    yaml.safe_dump(data, fh, sort_keys=False)

  with open(TEMPLATE_DIR / FINETUNE_TEMPLATE) as fh:
    cfg = yaml.safe_load(fh)
  cfg["defaults"][0]["data"] = data_name
  cfg["job"].update(
    device_type=args.device.upper(),
    run_dir=str(ft_dir / "runs"),
    run_name="finetune",
    timestamp_id=RUN_TIMESTAMP,
    logger=None,
    seed=args.seed,
  )
  cfg.update(
    base_model_name=model_name,
    epochs=args.epochs,
    steps=None,
    batch_size=args.batch_size,
    lr=args.lr,
    evaluate_every_n_steps=args.eval_every,
    # Checkpoints hold model + optimizer state (~3x model size); only the final one is kept.
    checkpoint_every_n_steps=10 ** 9,
  )
  for split in ("train_dataset", "val_dataset"):
    configs = cfg[split]["dataset_configs"]
    configs[TASK_NAME] = configs.pop("DATASET_NAME")
  for callback in cfg["runner"]["callbacks"]:
    if "max_saved_checkpoints" in callback:
      callback["max_saved_checkpoints"] = 1
  path = ft_dir / "finetune.yaml"
  with open(path, "w") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False)
  return path


def fairchem_command() -> list:
  """Command that launches the fairchem CLI from the same (uv) venv as this interpreter."""
  exe = Path(sys.executable).parent / "fairchem"
  if exe.exists():
    return [str(exe)]
  return [sys.executable, "-c", "from fairchem.core._cli import main; main()"]


def collect_checkpoint(ft_dir: Path, keep_checkpoints: bool) -> Path:
  """Move the final inference checkpoint to ``<ft_dir>/tuned.pt`` and prune the rest.

  Parameters
  ----------
  ft_dir : Path
    Fine-tune working directory.
  keep_checkpoints : bool
    If True, leave the full training checkpoints (resumable, several GB) in place
    and copy instead of move.

  Returns
  -------
  Path
    Path of the tuned inference checkpoint.
  """
  ckpt_dir = ft_dir / "runs" / RUN_TIMESTAMP / "checkpoints"
  final = ckpt_dir / "final" / "inference_ckpt.pt"
  if not final.exists():
    raise SystemExit(f"Training finished without {final}; check {ft_dir / 'runs'} logs")
  tuned = ft_dir / TUNED_CHECKPOINT
  if keep_checkpoints:
    shutil.copy2(final, tuned)
  else:
    shutil.move(str(final), tuned)
    shutil.rmtree(ckpt_dir)
  print(f"Tuned model: {tuned} ({tuned.stat().st_size / 1e9:.2f} GB)")
  return tuned


# ---------------------------------------------------------------- entry points

def _base_parser(description: str, default_device: str) -> argparse.ArgumentParser:
  """Arguments shared by all per-model scripts."""
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument("--labeled", type=Path, default=DEFAULT_LABELED, help="labeled extxyz from make_dataset.py label")
  parser.add_argument("--device", choices=["cpu", "cuda"], default=default_device)
  return parser


def run_zero_shot(model_name: str, model_dir: Path, default_device: str = DEFAULT_DEVICE, argv: list = None) -> None:
  """CLI: benchmark the pretrained model on every labeled frame.

  Parameters
  ----------
  model_name : str
    Pretrained UMA model name.
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
  parser = _base_parser(f"Zero-shot benchmark of pretrained {model_name} (task {TASK_NAME}) on CP2K labels.", default_device)
  parser.add_argument("--out", type=Path, default=model_dir / "zero_shot")
  args = parser.parse_args(argv)
  frames = load_labeled(args.labeled)
  print(f"{len(frames)} labeled frames from {args.labeled}; loading {model_name} on {args.device}")
  preds = predict(frames, make_calc(model_name, args.device))
  meta = {"model": model_name, "tuned": False, "task": TASK_NAME, "device": args.device, "labeled": str(args.labeled)}
  write_report(args.out, frames, preds, benchmark(frames, preds), meta)


def run_finetune(
  model_name: str,
  model_dir: Path,
  default_device: str = DEFAULT_DEVICE,
  cpu_ok: bool = True,
  argv: list = None,
) -> None:
  """CLI: fine-tune the pretrained model on the labeled frames, then benchmark it.

  Parameters
  ----------
  model_name : str
    Pretrained UMA model name.
  model_dir : Path
    ``models/<model_name>``; work goes to ``<model_dir>/finetune``.
  default_device : str, optional
    Device used when ``--device`` is not given.
  cpu_ok : bool, optional
    False for models too large to train on CPU; then ``--force`` is required.
  argv : list of str, optional
    Arguments (defaults to ``sys.argv[1:]``).

  Returns
  -------
  None
  """
  parser = _base_parser(f"Fine-tune {model_name} (task {TASK_NAME}) on CP2K labels and benchmark it.", default_device)
  parser.add_argument("--out", type=Path, default=model_dir / "finetune")
  parser.add_argument("--split", choices=["by-type", "random"], default="by-type",
                      help="by-type: rigid_scan = test, part of rattles = val (default)")
  parser.add_argument("--val-frac", type=float, default=0.2)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--regression", choices=["efs", "ef"], default="efs", help="train on energy+forces(+stress)")
  parser.add_argument("--epochs", type=int, default=100)
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--lr", type=float, default=4e-4)
  parser.add_argument("--eval-every", type=int, default=None, help="steps between validations (default: once per epoch)")
  parser.add_argument("--allow-tiny", action="store_true", help="smoke test: reuse train frames as val if a split is empty")
  parser.add_argument("--keep-checkpoints", action="store_true", help="keep full training checkpoints (several GB)")
  parser.add_argument("--prepare-only", action="store_true", help="write LMDBs and YAML, do not train")
  parser.add_argument("--force", action="store_true", help="train even where it is not recommended (large model on CPU)")
  args = parser.parse_args(argv)

  if args.device == "cpu" and not cpu_ok and not args.force and not args.prepare_only:
    raise SystemExit(f"{model_name} is too large to fine-tune on CPU (20+ GB RAM, days). Use --device cuda or --force.")
  ft_dir = args.out.resolve()
  if (ft_dir / "runs").exists():
    raise SystemExit(f"{ft_dir / 'runs'} exists; move it away or pass a new --out")
  mark_untracked(ft_dir)

  frames = load_labeled(args.labeled)
  splits = split_frames(frames, args.split, args.val_frac, args.seed)
  if not splits["val"] or not splits["train"]:
    if not args.allow_tiny:
      sizes = {k: len(v) for k, v in splits.items()}
      raise SystemExit(f"Empty split {sizes}; label more frames or use --allow-tiny")
    pool = splits["train"] or splits["val"] or splits["test"]
    splits["train"] = splits["train"] or pool
    splits["val"] = splits["val"] or pool
    print("WARNING: --allow-tiny reused frames across splits; metrics are not meaningful.")
  print("Split: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
  for name, part in splits.items():
    if part:
      (ft_dir / "split").mkdir(parents=True, exist_ok=True)
      write(ft_dir / "split" / f"{name}.extxyz", part)
  for name in ("train", "val"):
    write_aselmdb(splits[name], ft_dir / "lmdb" / name)

  force_rms, elem_refs = normalizers(splits["train"])
  if args.eval_every is None:
    args.eval_every = max(1, len(splits["train"]) // args.batch_size)
  config = build_config(ft_dir, args, model_name, force_rms, elem_refs)
  print(f"Wrote {config} (force RMS {force_rms:.4f} eV/A)")
  if args.prepare_only:
    print("Train with: " + " ".join(fairchem_command() + ["-c", str(config)]))
    return

  subprocess.run(fairchem_command() + ["-c", str(config)], check=True)
  tuned = collect_checkpoint(ft_dir, args.keep_checkpoints)
  calc = make_calc(tuned, args.device)
  meta = {"model": f"{model_name} fine-tuned", "tuned": True, "base_model": model_name, "task": TASK_NAME,
          "device": args.device, "labeled": str(args.labeled), "split": args.split, "epochs": args.epochs,
          "lr": args.lr, "regression": args.regression}
  for name, part in splits.items():
    if part:
      print(f"Benchmark on {name} split:")
      preds = predict(part, calc)
      write_report(ft_dir / "benchmark" / name, part, preds, benchmark(part, preds), {**meta, "eval_split": name})


def run_benchmark_tuned(model_name: str, model_dir: Path, default_device: str = DEFAULT_DEVICE, argv: list = None) -> None:
  """CLI: benchmark an existing fine-tuned checkpoint on a labeled file.

  Parameters
  ----------
  model_name : str
    Base UMA model name (for labels only).
  model_dir : Path
    ``models/<model_name>``; checkpoint defaults to ``<model_dir>/finetune/tuned.pt``.
  default_device : str, optional
    Device used when ``--device`` is not given.
  argv : list of str, optional
    Arguments (defaults to ``sys.argv[1:]``).

  Returns
  -------
  None
  """
  parser = _base_parser(f"Benchmark a fine-tuned {model_name} checkpoint on CP2K labels.", default_device)
  parser.add_argument("--checkpoint", type=Path, default=model_dir / "finetune" / TUNED_CHECKPOINT)
  parser.add_argument("--out", type=Path, default=model_dir / "finetune" / "benchmark" / "custom")
  args = parser.parse_args(argv)
  if not args.checkpoint.exists():
    raise SystemExit(f"{args.checkpoint} not found; run finetune.py first")
  frames = load_labeled(args.labeled)
  preds = predict(frames, make_calc(args.checkpoint, args.device))
  meta = {"model": f"{model_name} fine-tuned", "tuned": True, "checkpoint": str(args.checkpoint),
          "device": args.device, "labeled": str(args.labeled)}
  results = benchmark(frames, preds)
  write_report(args.out, frames, preds, results, meta)
  zero = model_dir / "zero_shot" / "results.json"
  if zero.exists():
    with open(zero) as fh:
      base = json.load(fh)["metrics"]["all"]
    print(f"zero-shot (all frames): F MAE {base['force_mae_eV_A']:.4f} eV/A, "
          f"dE MAE {base['rel_energy_mae_meV_per_atom']:.4f} meV/atom")
