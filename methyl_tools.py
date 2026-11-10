"""Geometry tools for generating methyl-rotor training configurations.

All functions take and return ASE ``Atoms`` objects and never modify their
input in place. Every generated frame carries metadata in ``atoms.info``
(``config_type``, ``phi_deg``, ...) so that it survives a round trip through
extended-XYZ files and can be filtered later.
"""

import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.io import write
from ase.io.jsonio import write_json
from ase.io.cif import parse_cif
from ase.neighborlist import NeighborList, natural_cutoffs, neighbor_list
from ase.optimize import BFGS

# Typical neutron-derived X-H bond lengths in Angstrom (Allen & Bruno,
# Acta Cryst. B 66, 380 (2010) give values close to these). X-ray structures
# place H too close to the heavy atom, so these are used to normalise them.
DEFAULT_XH_LENGTHS = {"C": 1.089, "N": 1.015, "O": 0.993}


@dataclass(frozen=True)
class MethylGroup:
  """Atom indices of one methyl rotor.

  Attributes
  ----------
  carbon : int
    Index of the methyl carbon.
  anchor : int
    Index of the heavy atom bonded to the carbon (the "X" in X-CH3).
    The rotation axis runs from this atom through the carbon.
  hydrogens : tuple of int
    Indices of the three methyl hydrogens.
  """

  carbon: int
  anchor: int
  hydrogens: tuple


def build_neighbor_list(atoms: Atoms, mult: float = 1.2) -> NeighborList:
  """Build a covalent-bond neighbour list for a periodic structure.

  Parameters
  ----------
  atoms : Atoms
    Structure to analyse. Shape of positions: (n_atoms, 3).
  mult : float, optional
    Multiplier on ASE natural covalent radii. 1.2 catches C-H bonds that
    are slightly long or short. Defaults to 1.2.

  Returns
  -------
  NeighborList
    Updated neighbour list (self-interaction excluded, both directions).
  """
  cutoffs = natural_cutoffs(atoms, mult=mult)
  nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
  nl.update(atoms)
  return nl


def find_methyl_groups(atoms: Atoms) -> list:
  """Find every methyl group: a carbon with exactly three H and one other neighbour.

  Parameters
  ----------
  atoms : Atoms
    Structure to search. Shape of positions: (n_atoms, 3).

  Returns
  -------
  list of MethylGroup
    One entry per methyl rotor found, in order of carbon index.
  """
  nl = build_neighbor_list(atoms)
  symbols = atoms.get_chemical_symbols()
  methyls = []
  for idx, sym in enumerate(symbols):
    if sym != "C":
      continue
    neighbors, _ = nl.get_neighbors(idx)
    h_idx = [int(j) for j in neighbors if symbols[j] == "H"]
    heavy_idx = [int(j) for j in neighbors if symbols[j] != "H"]
    if len(h_idx) == 3 and len(heavy_idx) == 1:
      methyls.append(MethylGroup(idx, heavy_idx[0], tuple(sorted(h_idx))))
  return methyls


def cif_conditions(path: str) -> dict:
  """Read measurement conditions from a CIF so unsuitable structures can be flagged.

  Parameters
  ----------
  path : str
    CIF file (first data block is used).

  Returns
  -------
  dict
    Keys ``block``, ``temperature_K``, ``pressure_kPa``, ``radiation``,
    ``h_treatment``; a value is None when the tag is absent.
  """
  block = next(iter(parse_cif(str(path))))

  def number(tag):
    value = block.get(tag)
    if value is None or str(value).strip() in ("?", "."):
      return None
    return float(re.sub(r"\(.*\)", "", str(value)))

  return {
    "block": block.name,
    "temperature_K": number("_diffrn_ambient_temperature") or number("_cell_measurement_temperature"),
    "pressure_kPa": number("_diffrn_ambient_pressure"),
    "radiation": block.get("_diffrn_radiation_type"),
    "h_treatment": block.get("_refine_ls_hydrogen_treatment"),
  }


def structure_warnings(conditions: dict, max_temperature: float = 150.0) -> list:
  """Reasons a CIF is a poor reference for a ~4 K, ambient-pressure tunneling measurement.

  Parameters
  ----------
  conditions : dict
    Output of ``cif_conditions``.
  max_temperature : float, optional
    Highest acceptable structure temperature in K; matches the Tier-1
    threshold in methyl_benchmark.xlsx. Defaults to 150.

  Returns
  -------
  list of str
    Human-readable warnings; empty if none apply.
  """
  warnings = []
  temp = conditions["temperature_K"]
  if temp is None:
    warnings.append("structure temperature not given")
  elif temp > max_temperature:
    warnings.append(f"structure measured at {temp:g} K (> {max_temperature:g} K)")
  pressure = conditions["pressure_kPa"]
  if pressure is not None and pressure > 200.0:
    warnings.append(f"structure measured at {pressure / 1e6:g} GPa, not ambient pressure")
  radiation = str(conditions["radiation"] or "").lower()
  if "neutron" not in radiation:
    warnings.append(f"radiation '{conditions['radiation']}': H positions are not from neutron data")
  return warnings


def _azimuths(atoms: Atoms, carbon: int, anchor: int, hydrogens: list) -> np.ndarray:
  """Angle of each H about the anchor->carbon axis, in degrees within [0, 360).

  Parameters
  ----------
  atoms : Atoms
    Periodic structure.
  carbon, anchor : int
    Methyl carbon and its heavy-atom neighbour.
  hydrogens : list of int
    H atoms bonded to ``carbon``.

  Returns
  -------
  np.ndarray
    Azimuths in degrees. Shape: (len(hydrogens),).
  """
  axis = atoms.get_distance(anchor, carbon, mic=True, vector=True)
  axis = axis / np.linalg.norm(axis)
  ch = np.array([atoms.get_distance(carbon, h, mic=True, vector=True) for h in hydrogens])
  # ch: (n_h, 3)
  perp = ch - np.outer(ch @ axis, axis)
  # perp: (n_h, 3), C-H vectors projected onto the plane normal to the axis
  ref = perp[0] / np.linalg.norm(perp[0])
  ortho = np.cross(axis, ref)
  return np.degrees(np.arctan2(perp @ ortho, perp @ ref)) % 360.0


def resolve_methyl_disorder(atoms: Atoms, max_sweeps: int = 10) -> Atoms:
  """Collapse two-orientation methyl disorder (6 half-occupied H) to one ordered orientation.

  Refinement programs often model a methyl as two staggered H triplets
  60 deg apart, each at 50 % occupancy (e.g. SHELX HFIX 123). Those positions
  are a modelling convention, not measured H positions, so the choice made
  here is only a starting guess: relax the H atoms afterwards
  (``relax_hydrogens``).

  For each disordered methyl the six H are sorted by azimuth and split into
  the two alternating triplets. Every methyl starts with the first triplet;
  then, sweep by sweep, each methyl switches to whichever triplet has the
  larger minimum distance to all other kept atoms, until nothing changes.

  Parameters
  ----------
  atoms : Atoms
    Structure as read from the CIF, including all partial-occupancy sites.
  max_sweeps : int, optional
    Maximum passes of the steric choice. Defaults to 10.

  Returns
  -------
  Atoms
    Copy with three H per methyl; ``info["methyl_disorder_resolved"]`` holds
    the number of methyls changed.

  Raises
  ------
  ValueError
    If a carbon has 4, 5 or more than 6 H, or a 6-H carbon has no single
    heavy neighbour, i.e. disorder this function does not understand.
  """
  nl = build_neighbor_list(atoms)
  symbols = atoms.get_chemical_symbols()
  groups = []
  # groups: list of (carbon, anchor, triplet_a, triplet_b); triplets are lists of 3 H indices
  for c in (i for i, s in enumerate(symbols) if s == "C"):
    neighbors, _ = nl.get_neighbors(c)
    h_idx = [int(j) for j in neighbors if symbols[j] == "H"]
    heavy = [int(j) for j in neighbors if symbols[j] != "H"]
    if len(h_idx) <= 3:
      continue
    if len(h_idx) != 6 or len(heavy) != 1:
      raise ValueError(
        f"C{c} has {len(h_idx)} H and {len(heavy)} heavy neighbours; resolve this disorder by hand")
    order = np.argsort(_azimuths(atoms, c, heavy[0], h_idx))
    ranked = [h_idx[k] for k in order]
    groups.append((c, heavy[0], ranked[0::2], ranked[1::2]))

  if not groups:
    return atoms.copy()

  choice = [0] * len(groups)

  def kept_indices():
    dropped = {h for (_, _, a, b), ch in zip(groups, choice) for h in (b if ch == 0 else a)}
    return np.array([i for i in range(len(atoms)) if i not in dropped])

  for _ in range(max_sweeps):
    changed = False
    for g, (c, anchor, trip_a, trip_b) in enumerate(groups):
      own = set(trip_a + trip_b + [c, anchor])
      others = np.array([i for i in kept_indices() if i not in own])
      # others: (n_others,), every kept atom outside this methyl's C, anchor and H
      score = []
      for trip in (trip_a, trip_b):
        d = np.concatenate([atoms.get_distances(h, others, mic=True) for h in trip])
        # d: (3 * n_others,)
        score.append(d.min())
      best = int(np.argmax(score)) if abs(score[0] - score[1]) > 1e-3 else choice[g]
      if best != choice[g]:
        choice[g], changed = best, True
    if not changed:
      break

  result = atoms[kept_indices()]
  result.info["methyl_disorder_resolved"] = len(groups)
  if "occupancy" in result.info:
    del result.info["occupancy"]  # occupancies no longer describe the ordered model
  return result


def check_structure(atoms: Atoms, min_distance: float = 0.9) -> None:
  """Fail loudly on overlapping atoms or methyls without exactly three H.

  Parameters
  ----------
  atoms : Atoms
    Structure to check (unit cell or supercell).
  min_distance : float, optional
    Smallest allowed interatomic distance in Angstrom. Defaults to 0.9,
    below every X-H bond but above overlapping disorder sites.

  Returns
  -------
  None

  Raises
  ------
  ValueError
    On any pair closer than ``min_distance`` or any carbon with more than
    three H.
  """
  i, j, d = neighbor_list("ijd", atoms, min_distance)
  if len(d):
    k = int(np.argmin(d))
    raise ValueError(
      f"{len(d) // 2} atom pairs closer than {min_distance} A (shortest {d[k]:.3f} A, "
      f"{atoms[i[k]].symbol}{i[k]}-{atoms[j[k]].symbol}{j[k]}); unresolved disorder?")
  nl = build_neighbor_list(atoms)
  symbols = atoms.get_chemical_symbols()
  for c in (n for n, s in enumerate(symbols) if s == "C"):
    neighbors, _ = nl.get_neighbors(c)
    n_h = sum(symbols[x] == "H" for x in neighbors)
    if n_h > 3:
      raise ValueError(f"C{c} has {n_h} H neighbours")


def write_atomic(path: Path, atoms: Atoms) -> None:
  """Write one structure to extxyz through a temporary file and a rename.

  Readers (and an interrupted writer) never see a half-written file.

  Parameters
  ----------
  path : Path
    Destination file.
  atoms : Atoms
    Structure to write; not modified.

  Returns
  -------
  None
  """
  path = Path(path)
  tmp = path.with_name(path.name + ".tmp")
  write(tmp, atoms, format="extxyz")
  os.replace(tmp, path)


def _dump_atomic(path: Path, data: tuple) -> None:
  """Write BFGS restart data (Hessian, last positions and forces) through a temporary file and a rename.

  Parameters
  ----------
  path : Path or None
    BFGS restart file; nothing is written if None.
  data : tuple
    ``(hessian, pos0, forces0, maxstep)`` as produced by ``BFGS.step``.

  Returns
  -------
  None
  """
  if path is None:
    return
  path = Path(path)
  tmp = path.with_name(path.name + ".tmp")
  with open(tmp, "w") as fd:
    write_json(fd, data)
  os.replace(tmp, path)


def _bfgs(atoms: Atoms, restart: Path = None) -> BFGS:
  """ASE ``BFGS`` logging to stdout whose restart file is written atomically."""
  opt = BFGS(atoms, logfile="-", restart=None if restart is None else str(restart))
  opt.dump = lambda data: _dump_atomic(opt.restart, data)
  return opt


def _make_bfgs(atoms: Atoms, hessian_file: Path, resuming: bool) -> BFGS:
  """Build the H-relaxation optimiser, restoring the Hessian from ``hessian_file`` if possible.

  Parameters
  ----------
  atoms : Atoms
    Constrained structure with a calculator attached (optimised in place).
  hessian_file : Path or None
    BFGS restart file, rewritten after every step; None disables it.
  resuming : bool
    True when continuing an earlier run; only then is a missing file reported.

  Returns
  -------
  BFGS
    Optimiser that logs to stdout.
  """
  if hessian_file is None:
    return _bfgs(atoms)
  hessian_file = Path(hessian_file)
  if not hessian_file.exists():
    if resuming:
      print(f"No BFGS Hessian file {hessian_file}; restarting BFGS from the checkpoint geometry.")
    return _bfgs(atoms, hessian_file)
  n_dof = 3 * len(atoms)
  try:
    opt = _bfgs(atoms, hessian_file)
    if opt.H.shape != (n_dof, n_dof):
      raise ValueError(f"Hessian shape {opt.H.shape}, expected {(n_dof, n_dof)}")
  except Exception as err:
    print(f"Could not restore the BFGS Hessian from {hessian_file} ({err}); "
          "restarting BFGS from the checkpoint geometry.")
    opt = _bfgs(atoms)
    opt.restart = str(hessian_file)
    return opt
  print(f"Restored the BFGS Hessian from {hessian_file}.")
  return opt


def relax_hydrogens(atoms: Atoms, calc, fmax: float = 0.02, max_steps: int = 500,
                    checkpoint: Path = None, progress_log: Path = None, hessian_file: Path = None,
                    start_step: int = 0) -> Atoms:
  """Optimise only the H positions; heavy atoms and the cell stay fixed.

  Use after ``resolve_methyl_disorder`` or ``normalize_xh_bonds``, because
  X-ray H positions (especially riding-model methyl H) carry little
  experimental information.

  With ``checkpoint``/``progress_log``/``hessian_file`` set, every optimiser step
  overwrites the checkpoint structure (atomically), appends one line to the
  progress log and saves the BFGS state, so an interrupted run can be resumed by
  passing the checkpoint geometry as ``atoms`` together with ``start_step``.

  Parameters
  ----------
  atoms : Atoms
    Structure to relax (the checkpoint geometry when resuming).
  calc : ase.calculators.calculator.Calculator
    Energy/force calculator.
  fmax : float, optional
    Force threshold on H in eV/Angstrom. Defaults to 0.02.
  max_steps : int, optional
    Maximum BFGS steps, counted across restarts. Defaults to 500.
  checkpoint : Path, optional
    extxyz file overwritten with the current structure (energy, forces and
    ``info["h_relax_step"]``) after every step.
  progress_log : Path, optional
    Tab-separated log, one line per step: step, time, energy (eV),
    max H force (eV/A), fmax (eV/A). Appended to, never truncated.
  hessian_file : Path, optional
    ASE BFGS restart file. Read if it exists (falls back to a fresh Hessian if
    it cannot be used), rewritten after every step.
  start_step : int, optional
    Number of steps already done by earlier runs. Defaults to 0.

  Returns
  -------
  Atoms
    Relaxed copy with constraints removed; ``info["h_relaxed"]`` is True and
    ``info["h_relax_converged"]`` records convergence.
  """
  result = atoms.copy()
  symbols = result.get_chemical_symbols()
  heavy = [i for i, s in enumerate(symbols) if s != "H"]
  is_h = np.array([s == "H" for s in symbols])
  # is_h: (n_atoms,)
  result.set_constraint(FixAtoms(indices=heavy))
  result.calc = calc
  opt = _make_bfgs(result, hessian_file, resuming=start_step > 0)

  if progress_log is not None:
    progress_log = Path(progress_log)
    with open(progress_log, "a") as fh:
      if progress_log.stat().st_size == 0:
        fh.write("# step\ttime\tenergy_eV\tmax_H_force_eV_A\tfmax_eV_A\n")
      if start_step > 0:
        fh.write(f"# restart at step {start_step} {datetime.now().isoformat(timespec='seconds')}\n")

  def record() -> None:
    """Observer: checkpoint the structure and log one progress line."""
    step = start_step + opt.nsteps
    energy = result.get_potential_energy()
    forces = result.get_forces()
    # forces: (n_atoms, 3), heavy-atom rows zeroed by FixAtoms
    f_h = float(np.linalg.norm(forces[is_h], axis=1).max())
    if checkpoint is not None:
      snap = result.copy()
      snap.set_constraint()
      snap.calc = SinglePointCalculator(snap, energy=energy, forces=result.get_forces(apply_constraint=False))
      snap.info.update(h_relax_step=step, h_relax_fmax=fmax)
      write_atomic(checkpoint, snap)
    if progress_log is not None:
      with open(progress_log, "a") as fh:
        fh.write(f"{step}\t{datetime.now().isoformat(timespec='seconds')}\t{energy:.6f}\t{f_h:.5f}\t{fmax:g}\n")

  if checkpoint is not None or progress_log is not None:
    opt.attach(record)
  converged = opt.run(fmax=fmax, steps=max(max_steps - start_step, 0))
  relaxed = result.copy()
  relaxed.set_constraint()
  relaxed.info.update(h_relaxed=True, h_relax_converged=bool(converged))
  return relaxed


def normalize_xh_bonds(atoms: Atoms, lengths: dict = None) -> Atoms:
  """Move every H along its bond so X-H lengths match neutron-like values.

  Parameters
  ----------
  atoms : Atoms
    Structure, typically read from an X-ray CIF. Shape of positions:
    (n_atoms, 3).
  lengths : dict, optional
    Map from heavy-atom symbol to target X-H length in Angstrom.
    Defaults to ``DEFAULT_XH_LENGTHS``.

  Returns
  -------
  Atoms
    Copy of ``atoms`` with H positions adjusted; heavy atoms untouched.
  """
  lengths = DEFAULT_XH_LENGTHS if lengths is None else lengths
  result = atoms.copy()
  nl = build_neighbor_list(result)
  symbols = result.get_chemical_symbols()
  positions = result.get_positions()
  # positions: (n_atoms, 3)
  for h in (i for i, s in enumerate(symbols) if s == "H"):
    neighbors, _ = nl.get_neighbors(h)
    heavy = [j for j in neighbors if symbols[j] in lengths]
    if len(heavy) != 1:
      continue
    x = int(heavy[0])
    bond = result.get_distance(x, h, mic=True, vector=True)
    # bond: (3,), vector from X to H under minimum image
    positions[h] = positions[x] + bond / np.linalg.norm(bond) * lengths[symbols[x]]
  result.set_positions(positions)
  result.wrap()
  return result


def supercell_repeats(cell: np.ndarray, min_length: float) -> tuple:
  """Number of repeats per lattice vector so every perpendicular width >= min_length.

  Parameters
  ----------
  cell : np.ndarray
    Lattice vectors as rows. Shape: (3, 3).
  min_length : float
    Minimum distance in Angstrom between opposite faces of the supercell,
    i.e. a lower bound on the rotor-image separation.

  Returns
  -------
  tuple of int
    Repeats (n_a, n_b, n_c).
  """
  volume = abs(np.linalg.det(cell))
  repeats = []
  for i in range(3):
    j, k = (i + 1) % 3, (i + 2) % 3
    width = volume / np.linalg.norm(np.cross(cell[j], cell[k]))
    repeats.append(max(1, math.ceil(min_length / width)))
  return tuple(repeats)


def make_supercell(atoms: Atoms, min_length: float = 10.0) -> Atoms:
  """Repeat a unit cell until rotor images are at least ``min_length`` apart.

  Parameters
  ----------
  atoms : Atoms
    Unit cell. Shape of positions: (n_atoms, 3).
  min_length : float, optional
    Minimum perpendicular cell width in Angstrom. Defaults to 10.0.

  Returns
  -------
  Atoms
    Supercell with n_atoms * n_a * n_b * n_c atoms; ``info["repeats"]`` is set.
  """
  repeats = supercell_repeats(np.array(atoms.get_cell()), min_length)
  supercell = atoms.repeat(repeats)
  supercell.info["repeats"] = repeats
  return supercell


def rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
  """Rodrigues rotation matrix about a unit-normalised axis.

  Parameters
  ----------
  axis : np.ndarray
    Rotation axis (any length > 0). Shape: (3,).
  angle_deg : float
    Right-handed rotation angle in degrees.

  Returns
  -------
  np.ndarray
    Rotation matrix. Shape: (3, 3).
  """
  u = axis / np.linalg.norm(axis)
  theta = math.radians(angle_deg)
  k = np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])
  # k: (3, 3), cross-product matrix of u
  return np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)


def rotate_methyl(atoms: Atoms, methyl: MethylGroup, delta_deg: float) -> Atoms:
  """Rotate one methyl's hydrogens about the anchor->carbon axis.

  Only the three H atoms move, so bond lengths and angles are preserved
  exactly. Periodic images are handled with the minimum-image convention.

  Parameters
  ----------
  atoms : Atoms
    Structure containing the methyl. Shape of positions: (n_atoms, 3).
  methyl : MethylGroup
    The rotor to turn.
  delta_deg : float
    Rotation relative to the current orientation, in degrees.

  Returns
  -------
  Atoms
    Rotated copy, wrapped back into the cell.
  """
  result = atoms.copy()
  positions = result.get_positions()
  # positions: (n_atoms, 3)
  axis = result.get_distance(methyl.anchor, methyl.carbon, mic=True, vector=True)
  rot = rotation_matrix(axis, delta_deg)
  # rot: (3, 3)
  c_pos = positions[methyl.carbon]
  for h in methyl.hydrogens:
    ch = result.get_distance(methyl.carbon, h, mic=True, vector=True)
    positions[h] = c_pos + rot @ ch
  result.set_positions(positions)
  result.wrap()
  return result


def rigid_scan(atoms: Atoms, methyl: MethylGroup, angles_deg: np.ndarray) -> list:
  """Frozen-environment scan: rotate one methyl, nothing else moves.

  Parameters
  ----------
  atoms : Atoms
    Reference (equilibrium) supercell.
  methyl : MethylGroup
    The rotor to scan.
  angles_deg : np.ndarray
    Rotation angles relative to the reference orientation. Shape: (n_angles,).

  Returns
  -------
  list of Atoms
    One frame per angle, length n_angles, tagged with ``config_type="rigid_scan"``
    and ``phi_deg``.
  """
  frames = []
  for phi in angles_deg:
    frame = rotate_methyl(atoms, methyl, float(phi))
    frame.info.update(config_type="rigid_scan", phi_deg=float(phi), rotor_c=methyl.carbon)
    frames.append(frame)
  return frames


def atoms_within(atoms: Atoms, center: int, radius: float) -> np.ndarray:
  """Indices of atoms within ``radius`` of ``center`` (minimum image).

  Parameters
  ----------
  atoms : Atoms
    Periodic structure.
  center : int
    Index of the central atom.
  radius : float
    Cutoff in Angstrom.

  Returns
  -------
  np.ndarray
    Integer indices, including ``center``. Shape: (n_inside,).
  """
  dists = atoms.get_distances(center, np.arange(len(atoms)), mic=True)
  # dists: (n_atoms,)
  return np.flatnonzero(dists <= radius)


def relaxed_scan(
  atoms: Atoms,
  methyl: MethylGroup,
  angles_deg: np.ndarray,
  calc,
  relax_radius: float = 6.0,
  fmax: float = 0.02,
  max_steps: int = 300,
  keep_every: int = 5,
) -> tuple:
  """Constrained scan: methyl held rigid at each angle, a shell of neighbours relaxes.

  The methyl (C, its 3 H, and the anchor atom) is fixed, which fixes the
  torsion angle without needing a fourth atom to define a dihedral. Atoms
  within ``relax_radius`` of the methyl carbon relax; everything else and the
  cell stay frozen. Set ``relax_radius`` large to relax the whole supercell.

  Parameters
  ----------
  atoms : Atoms
    Reference supercell.
  methyl : MethylGroup
    The rotor to scan.
  angles_deg : np.ndarray
    Rotation angles relative to the reference. Shape: (n_angles,).
  calc : ase.calculators.calculator.Calculator
    Energy/force calculator (an MLIP for cheap previews, CP2K for references).
  relax_radius : float, optional
    Radius in Angstrom around the methyl carbon that is allowed to relax.
    Defaults to 6.0.
  fmax : float, optional
    Force convergence criterion in eV/Angstrom. Defaults to 0.02.
  max_steps : int, optional
    Maximum optimiser steps per angle. Defaults to 300.
  keep_every : int, optional
    Keep every k-th intermediate optimiser frame as extra training data.
    Defaults to 5.

  Returns
  -------
  tuple of (list of Atoms, list of Atoms)
    (relaxed endpoints, length n_angles; intermediate frames for training).
  """
  fixed_rotor = {methyl.carbon, methyl.anchor, *methyl.hydrogens}
  mobile = set(atoms_within(atoms, methyl.carbon, relax_radius).tolist()) - fixed_rotor
  frozen = [i for i in range(len(atoms)) if i not in mobile]

  endpoints, intermediates = [], []
  for phi in angles_deg:
    frame = rotate_methyl(atoms, methyl, float(phi))
    frame.set_constraint(FixAtoms(indices=frozen))
    frame.calc = calc
    opt = BFGS(frame, logfile=None)
    for step, _ in enumerate(opt.irun(fmax=fmax, steps=max_steps)):
      if step > 0 and step % keep_every == 0:
        snap = frame.copy()
        snap.set_constraint()
        snap.info.update(config_type="relax_traj", phi_deg=float(phi), rotor_c=methyl.carbon)
        intermediates.append(snap)
    final = frame.copy()
    final.set_constraint()
    final.info.update(
      config_type="relaxed_scan",
      phi_deg=float(phi),
      rotor_c=methyl.carbon,
      relax_radius=relax_radius,
      e_model=float(frame.get_potential_energy()),
    )
    endpoints.append(final)
  return endpoints, intermediates


def rattled_frames(atoms: Atoms, n_frames: int, stdev: float = 0.05, seed: int = 0) -> list:
  """Random Gaussian displacements of all atoms (cheap stand-in for MD snapshots).

  Parameters
  ----------
  atoms : Atoms
    Reference structure.
  n_frames : int
    Number of frames to generate.
  stdev : float, optional
    Standard deviation of each Cartesian displacement in Angstrom.
    Defaults to 0.05.
  seed : int, optional
    Base random seed; frame k uses seed + k. Defaults to 0.

  Returns
  -------
  list of Atoms
    Length n_frames, tagged ``config_type="rattle"``.
  """
  frames = []
  for k in range(n_frames):
    frame = atoms.copy()
    frame.rattle(stdev=stdev, seed=seed + k)
    frame.wrap()
    frame.info.update(config_type="rattle", rattle_stdev=stdev)
    frames.append(frame)
  return frames


def strained_frames(atoms: Atoms, volume_strains: np.ndarray) -> list:
  """Isotropically scaled cells with fractional coordinates held fixed.

  Simple version: covalent bonds stretch by the linear strain too (about
  0.3-1 % for +/-1-3 % volume). A later refinement is to scale molecular
  centres of mass only and keep each molecule rigid.

  Parameters
  ----------
  atoms : Atoms
    Reference structure.
  volume_strains : np.ndarray
    Fractional volume changes, e.g. [-0.03, 0.03]. Shape: (n_strains,).

  Returns
  -------
  list of Atoms
    Length n_strains, tagged ``config_type="strain"``.
  """
  frames = []
  for dv in volume_strains:
    frame = atoms.copy()
    scale = (1.0 + float(dv)) ** (1.0 / 3.0)
    frame.set_cell(np.array(atoms.get_cell()) * scale, scale_atoms=True)
    frame.info.update(config_type="strain", volume_strain=float(dv))
    frames.append(frame)
  return frames
