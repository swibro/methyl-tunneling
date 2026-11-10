"""Command-line driver for building a methyl-rotor training set.

Subcommands, run in order:

  smoke     Check that ASE can drive CP2K (water molecule in a box).
  inspect   Read a CIF, flag unsuitable conditions, resolve methyl disorder,
            normalise X-H bonds, list methyl groups, write a supercell.
  relax-h   Relax only the H atoms of the supercell with CP2K (recommended
            after inspect, since X-ray/riding-model H positions are guesses).
            Checkpoints every step; resume with --restart.
  generate  Build unlabeled frames (rigid scan, rattles, strains) for one rotor.
  label     Run CP2K single points on every frame; resumable.

Examples
--------
  python make_dataset.py smoke
  python make_dataset.py inspect CH3I.cif --min-length 10
  python make_dataset.py relax-h work/supercell.extxyz --fmax 0.005
  python make_dataset.py relax-h work/supercell.extxyz --fmax 0.005 --restart
  python make_dataset.py generate work/supercell_hrelaxed.extxyz --rotor 0
  python make_dataset.py label work/unlabeled.extxyz --out work/labeled.extxyz
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

import methyl_tools as mt
from cp2k_setup import make_cp2k_calc


def cmd_smoke(args: argparse.Namespace) -> None:
  """Run one CP2K PBE-D3(BJ) single point on a water molecule and print the results.

  Parameters
  ----------
  args : argparse.Namespace
    Parsed arguments; uses ``cp2k_command``.

  Returns
  -------
  None
  """
  water = molecule("H2O", vacuum=4.0, pbc=True)
  with make_cp2k_calc(label="smoke", command=args.cp2k_command) as calc:
    water.calc = calc
    energy = water.get_potential_energy()
    forces = water.get_forces()
    # forces: (3, 3)
  print(f"Energy: {energy:.6f} eV")
  print(f"Max |F|: {np.abs(forces).max():.4f} eV/A")
  print("CP2K is reachable from ASE.")


def cmd_inspect(args: argparse.Namespace) -> None:
  """Clean a CIF, report its methyl groups, and write unit cell + supercell.

  Parameters
  ----------
  args : argparse.Namespace
    Parsed arguments; uses ``cif``, ``min_length``, ``workdir``.

  Returns
  -------
  None
  """
  workdir = Path(args.workdir)
  workdir.mkdir(parents=True, exist_ok=True)
  conditions = mt.cif_conditions(args.cif)
  print(f"CIF block {conditions['block']}: T = {conditions['temperature_K']} K, "
        f"P = {conditions['pressure_kPa']} kPa, radiation = {conditions['radiation']}")
  for warning in mt.structure_warnings(conditions):
    print(f"  WARNING: {warning}")

  unit = read(args.cif)
  print(f"Read {len(unit)} atoms, formula {unit.get_chemical_formula()}")
  print(f"Cell lengths/angles: {np.round(unit.cell.cellpar(), 3)}")

  unit = mt.resolve_methyl_disorder(unit)
  n_fixed = unit.info.get("methyl_disorder_resolved", 0)
  if n_fixed:
    print(f"Resolved two-orientation disorder on {n_fixed} methyl(s); "
          f"now {len(unit)} atoms, {unit.get_chemical_formula()}. Run relax-h next.")
  unit = mt.normalize_xh_bonds(unit)
  mt.check_structure(unit)
  methyls = mt.find_methyl_groups(unit)
  print(f"Methyl groups in unit cell: {len(methyls)}")
  for k, m in enumerate(methyls):
    print(f"  [{k}] C={m.carbon} anchor={unit[m.anchor].symbol}{m.anchor} H={m.hydrogens}")

  supercell = mt.make_supercell(unit, min_length=args.min_length)
  print(f"Supercell repeats {supercell.info['repeats']} -> {len(supercell)} atoms")
  write(workdir / "unit_cell_clean.extxyz", unit)
  write(workdir / "supercell.extxyz", supercell)
  print(f"Wrote {workdir}/unit_cell_clean.extxyz and supercell.extxyz")
  print("Next: open them in `ase gui` and check the H positions look sensible.")


def cmd_generate(args: argparse.Namespace) -> None:
  """Generate unlabeled configurations around one methyl rotor.

  Parameters
  ----------
  args : argparse.Namespace
    Parsed arguments; uses ``supercell``, ``rotor``, ``step``, ``n_rattle``,
    ``rattle_stdev``, ``workdir``.

  Returns
  -------
  None
  """
  supercell = read(args.supercell)
  mt.check_structure(supercell)
  if not supercell.info.get("h_relaxed"):
    print("WARNING: H positions not relaxed (run relax-h first for production data).")
  methyls = mt.find_methyl_groups(supercell)
  if not methyls:
    raise SystemExit("No methyl groups found (each needs exactly 3 H and 1 heavy neighbour).")
  if not 0 <= args.rotor < len(methyls):
    raise SystemExit(f"--rotor must be 0..{len(methyls) - 1}")
  methyl = methyls[args.rotor]
  print(f"Scanning rotor C={methyl.carbon} (of {len(methyls)} in supercell)")

  angles = np.arange(0.0, 120.0 + 1e-9, args.step)
  # angles: (n_angles,) ; 0-120 deg covers one period of a C3 rotor
  frames = [Atoms(supercell)]
  frames[0].info.update(config_type="reference")
  frames += mt.rigid_scan(supercell, methyl, angles)
  frames += mt.rattled_frames(supercell, args.n_rattle, stdev=args.rattle_stdev)
  frames += mt.strained_frames(supercell, np.array([-0.03, -0.015, 0.015, 0.03]))
  for k, frame in enumerate(frames):
    frame.info["frame_id"] = k

  out = Path(args.workdir) / "unlabeled.extxyz"
  write(out, frames)
  counts = {}
  for frame in frames:
    counts[frame.info["config_type"]] = counts.get(frame.info["config_type"], 0) + 1
  print(f"Wrote {len(frames)} frames to {out}: {counts}")


def relax_files(out: Path) -> dict:
  """Checkpoint, progress-log and BFGS-state paths that belong to a relax-h output file.

  Parameters
  ----------
  out : Path
    relax-h output, e.g. ``work/supercell_hrelaxed.extxyz``.

  Returns
  -------
  dict
    "checkpoint" (``<stem>.ckpt.extxyz``), "progress" (``<stem>.progress.log``)
    and "hessian" (``<stem>.bfgs.json``) next to ``out``.
  """
  out = Path(out)
  stem = out.name[:-len(out.suffix)] if out.suffix else out.name
  return {
    "checkpoint": out.with_name(f"{stem}.ckpt.extxyz"),
    "progress": out.with_name(f"{stem}.progress.log"),
    "hessian": out.with_name(f"{stem}.bfgs.json"),
  }


def load_relax_checkpoint(path: Path, supercell: Atoms) -> tuple:
  """Read a relax-h checkpoint and check that it belongs to ``supercell``.

  Parameters
  ----------
  path : Path
    Checkpoint extxyz written by ``methyl_tools.relax_hydrogens``.
  supercell : Atoms
    Original relax-h input; not modified.

  Returns
  -------
  tuple of (Atoms, int, float)
    Copy of ``supercell`` with the checkpoint positions, the number of steps
    already done, and the max H force at the checkpoint (eV/A).
  """
  ckpt = read(path)
  if ckpt.get_chemical_symbols() != supercell.get_chemical_symbols():
    raise SystemExit(f"{path} has different atoms than the input; delete it to start over.")
  is_h = np.array([s == "H" for s in supercell.get_chemical_symbols()])
  # is_h: (n_atoms,)
  same_cell = np.allclose(ckpt.cell.array, supercell.cell.array, atol=1e-6)
  same_heavy = np.allclose(ckpt.positions[~is_h], supercell.positions[~is_h], atol=1e-6)
  if not (same_cell and same_heavy):
    raise SystemExit(f"{path} does not match the input (cell or heavy atoms differ); delete it to start over.")
  start = supercell.copy()
  start.set_positions(ckpt.get_positions())
  forces = ckpt.get_forces()
  # forces: (n_atoms, 3)
  f_h = float(np.linalg.norm(forces[is_h], axis=1).max())
  return start, int(ckpt.info["h_relax_step"]), f_h


def cmd_relax_h(args: argparse.Namespace) -> None:
  """Relax only the H atoms of a supercell with CP2K and write the result.

  Every optimiser step is checkpointed (see ``relax_files``); ``--restart``
  resumes from the checkpoint, restoring the BFGS Hessian when possible.

  Parameters
  ----------
  args : argparse.Namespace
    Parsed arguments; uses ``supercell``, ``out``, ``fmax``, ``cp2k_command``,
    ``cutoff``, ``restart``.

  Returns
  -------
  None
  """
  supercell = read(args.supercell)
  mt.check_structure(supercell)
  files = relax_files(args.out)
  start, start_step = supercell, 0
  if args.restart:
    if not files["checkpoint"].exists():
      raise SystemExit(f"--restart: no checkpoint {files['checkpoint']}; run without --restart to start fresh.")
    start, start_step, f_h = load_relax_checkpoint(files["checkpoint"], supercell)
    print(f"Resuming from {files['checkpoint']}: step {start_step}, max H force {f_h:.4f} eV/A "
          f"(target {args.fmax} eV/A). The first step recomputes the checkpoint geometry.")
  else:
    leftovers = [str(p) for p in files.values() if p.exists()]
    if leftovers:
      raise SystemExit("Found files from an earlier relax-h run:\n  " + "\n  ".join(leftovers)
                       + "\nResume with --restart, or delete them to start over.")
  with make_cp2k_calc(label="relax_h", command=args.cp2k_command, cutoff_ry=args.cutoff) as calc:
    relaxed = mt.relax_hydrogens(start, calc, fmax=args.fmax, checkpoint=files["checkpoint"],
                                 progress_log=files["progress"], hessian_file=files["hessian"],
                                 start_step=start_step)
  shift = np.linalg.norm(relaxed.get_positions() - supercell.get_positions(), axis=1)
  # shift: (n_atoms,), Angstrom moved by each atom
  print(f"Max H displacement {shift.max():.3f} A; converged = {relaxed.info['h_relax_converged']}")
  relaxed.info["repeats"] = supercell.info.get("repeats")
  write(args.out, relaxed)
  print(f"Wrote {args.out}. Use it as the input to generate.")


def labeled_ids(path: Path) -> set:
  """Frame IDs already present in a labeled extxyz file (for resuming).

  Parameters
  ----------
  path : Path
    Labeled output file; may not exist yet.

  Returns
  -------
  set of int
    ``frame_id`` values already computed.
  """
  if not path.exists():
    return set()
  return {int(a.info["frame_id"]) for a in read(path, index=":")}


def cmd_label(args: argparse.Namespace) -> None:
  """Compute CP2K energy, forces and stress for every unlabeled frame.

  Results are appended one frame at a time, so an interrupted run can be
  restarted with the same command and will skip finished frames.

  Parameters
  ----------
  args : argparse.Namespace
    Parsed arguments; uses ``unlabeled``, ``out``, ``cp2k_command``,
    ``cutoff``. Each labeled frame also gets ``info["labeled_at"]`` (ISO time)
    and ``info["label_seconds"]`` (wall time of the CP2K calculation).

  Returns
  -------
  None
  """
  frames = read(args.unlabeled, index=":")
  out = Path(args.out)
  done = labeled_ids(out)
  todo = [f for f in frames if int(f.info["frame_id"]) not in done]
  print(f"{len(done)} frames already labeled, {len(todo)} to go")

  with make_cp2k_calc(label="label", command=args.cp2k_command, cutoff_ry=args.cutoff) as calc:
    for frame in todo:
      frame.calc = calc
      t0 = time.perf_counter()
      energy = frame.get_potential_energy()
      forces = frame.get_forces()
      # forces: (n_atoms, 3)
      stress = frame.get_stress()
      # stress: (6,) Voigt order, eV/A^3
      seconds = time.perf_counter() - t0
      labeled = frame.copy()
      labeled.info.update(labeled_at=datetime.now().isoformat(timespec="seconds"), label_seconds=round(seconds, 1))
      labeled.calc = SinglePointCalculator(labeled, energy=energy, forces=forces, stress=stress)
      write(out, labeled, append=True)
      print(f"  frame {frame.info['frame_id']:4d} {frame.info['config_type']:12s} E={energy:.5f} eV ({seconds:.0f} s)")


def build_parser() -> argparse.ArgumentParser:
  """Define the command-line interface.

  Returns
  -------
  argparse.ArgumentParser
    Parser with smoke / inspect / relax-h / generate / label subcommands.
  """
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--workdir", default="work")
  parser.add_argument("--cp2k-command", default=None, help='e.g. "mpirun -np 8 cp2k.psmp -s"')
  sub = parser.add_subparsers(dest="cmd", required=True)

  sub.add_parser("smoke")

  p_inspect = sub.add_parser("inspect")
  p_inspect.add_argument("cif")
  p_inspect.add_argument("--min-length", type=float, default=10.0)

  p_relax = sub.add_parser("relax-h")
  p_relax.add_argument("supercell")
  p_relax.add_argument("--out", default="work/supercell_hrelaxed.extxyz")
  p_relax.add_argument("--fmax", type=float, default=0.02, help="eV/A")
  p_relax.add_argument("--cutoff", type=float, default=500.0, help="density cutoff, Ry")
  p_relax.add_argument("--restart", action="store_true", help="resume from the checkpoint next to --out")

  p_gen = sub.add_parser("generate")
  p_gen.add_argument("supercell")
  p_gen.add_argument("--rotor", type=int, default=0)
  p_gen.add_argument("--step", type=float, default=10.0, help="scan step in degrees")
  p_gen.add_argument("--n-rattle", type=int, default=20)
  p_gen.add_argument("--rattle-stdev", type=float, default=0.05)

  p_label = sub.add_parser("label")
  p_label.add_argument("unlabeled")
  p_label.add_argument("--out", default="work/labeled.extxyz")
  p_label.add_argument("--cutoff", type=float, default=500.0, help="density cutoff, Ry")
  return parser


def main() -> None:
  """Parse arguments and dispatch to the chosen subcommand."""
  args = build_parser().parse_args()
  commands = {
    "smoke": cmd_smoke, "inspect": cmd_inspect, "relax-h": cmd_relax_h,
    "generate": cmd_generate, "label": cmd_label,
  }
  commands[args.cmd](args)


if __name__ == "__main__":
  main()
