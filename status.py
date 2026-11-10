"""Read-only status report for the methyl-rotor pipeline.

Checks each step (smoke -> inspect -> relax-h -> generate -> label) for missing,
stale (output older than its input) or inconsistent files, shows relax-h and
label progress, running jobs, recently written CP2K output and model benchmarks,
and ends with the one command to run next. Never writes anything.

Examples
--------
  python status.py
  python status.py --watch 60     # redraw every 60 s, Ctrl-C to stop
"""

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from ase.io import iread, read

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
SMOKE_OUT = ROOT / "smoke.out"
SUPERCELL = WORK / "supercell.extxyz"
HRELAXED = WORK / "supercell_hrelaxed.extxyz"
RELAX_CKPT = WORK / "supercell_hrelaxed.ckpt.extxyz"
RELAX_LOG = WORK / "supercell_hrelaxed.progress.log"
UNLABELED = WORK / "unlabeled.extxyz"
LABELED = WORK / "labeled.extxyz"
MODELS = ROOT / "models"
# Model folder patterns and the venv each family runs in (mace-torch and fairchem need
# incompatible e3nn versions, so they cannot share one venv).
MODEL_FAMILIES = (("uma-*", "menv"), ("mace-off*", "menv-mace"))
RELAX_FMAX = 0.005           # recommended relax-h --fmax, eV/A
POS_TOL = 1e-5               # Angstrom; "same geometry" tolerance
RECENT_SECONDS = 3600        # root *.out/*.log newer than this are listed
ACTIVE_SECONDS = 600         # a CP2K .out written this recently counts as running
MIN_SCAN_ANGLES = 5          # fewer rigid_scan angles cannot be fitted with V3, V6, phi0
SCF_LINE = re.compile(r"^\s+\d+\s+[A-Za-z].*?\s[\d.]+E[+-]\d+")
OK, BAD, WARN = "✓", "✗", "⚠"


# ---------------------------------------------------------------- helpers

def mtime(path: Path) -> float:
  """Modification time of ``path`` (s since epoch), or 0.0 if it does not exist."""
  return path.stat().st_mtime if path.exists() else 0.0


def fmt_time(t: float) -> str:
  """Format an epoch time as ``Oct 15 16:11``."""
  return datetime.fromtimestamp(t).strftime("%b %d %H:%M")


def fmt_age(seconds: float) -> str:
  """Format a duration as e.g. ``3h05m``, ``12m`` or ``40s``."""
  seconds = int(seconds)
  if seconds >= 3600:
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"
  if seconds >= 60:
    return f"{seconds // 60}m"
  return f"{seconds}s"


def rel(path: Path) -> str:
  """Path relative to the repo root when possible."""
  try:
    return str(path.relative_to(ROOT))
  except ValueError:
    return str(path)


def read_one(path: Path):
  """Read the first frame of an extxyz file; None if missing or unreadable."""
  try:
    return read(path)
  except Exception:
    return None


def read_frames(path: Path) -> tuple:
  """Read all frames, keeping those read before any error (file may be mid-append).

  Parameters
  ----------
  path : Path
    extxyz file.

  Returns
  -------
  tuple of (list of Atoms, str or None)
    Frames read, and the error message if reading stopped early.
  """
  frames = []
  try:
    for atoms in iread(path, index=":"):
      frames.append(atoms)
  except Exception as err:
    return frames, f"{type(err).__name__}: {err}"
  return frames, None


def same_positions(a, b, mask=None) -> bool:
  """True if two structures have the same symbols, cell and positions (within ``POS_TOL``).

  Parameters
  ----------
  a, b : Atoms
    Structures to compare; not modified.
  mask : np.ndarray, optional
    Boolean atom mask; only these positions are compared. Shape: (n_atoms,).

  Returns
  -------
  bool
  """
  if a is None or b is None or len(a) != len(b) or a.get_chemical_symbols() != b.get_chemical_symbols():
    return False
  if not np.allclose(a.cell.array, b.cell.array, atol=POS_TOL):
    return False
  pa, pb = a.get_positions(), b.get_positions()
  # pa, pb: (n_atoms, 3)
  if mask is not None:
    pa, pb = pa[mask], pb[mask]
  return bool(np.allclose(pa, pb, atol=POS_TOL))


# ---------------------------------------------------------------- pipeline checks

class Step:
  """Status of one pipeline step.

  Attributes
  ----------
  name : str
    Subcommand name.
  path : Path
    Main output file.
  state : str
    "ok", "missing", "stale", "bad" (inconsistent) or "partial".
  notes : list of str
    Extra lines printed under the step.
  """

  def __init__(self, name: str, path: Path):
    self.name, self.path, self.state, self.notes = name, path, "ok", []

  def mark(self, state: str, note: str = None) -> None:
    """Downgrade the state (never upgrade) and add a note."""
    order = ["ok", "partial", "stale", "bad", "missing"]
    if order.index(state) > order.index(self.state):
      self.state = state
    if note:
      self.notes.append(note)

  def line(self) -> str:
    """One checklist line: symbol, step name, file and mtime."""
    sym = {"ok": OK, "missing": BAD, "bad": BAD}.get(self.state, WARN)
    when = fmt_time(mtime(self.path)) if self.path.exists() else "-"
    tag = "" if self.state == "ok" else f"  [{self.state.upper()}]"
    return f"  {sym} {self.name:9s} {rel(self.path):36s} {when}{tag}"


def check_pipeline() -> dict:
  """Run every consistency check on the work files.

  Returns
  -------
  dict
    "steps" (dict name -> Step), "relax" (relax-h progress dict or None),
    "label" (label progress dict or None).
  """
  steps = {n: Step(n, p) for n, p in [("smoke", SMOKE_OUT), ("inspect", SUPERCELL), ("relax-h", HRELAXED),
                                        ("generate", UNLABELED), ("label", LABELED)]}

  # smoke
  s = steps["smoke"]
  if not SMOKE_OUT.exists():
    s.mark("missing")
  elif "ENERGY|" not in SMOKE_OUT.read_text(errors="replace"):
    s.mark("bad", "smoke.out has no ENERGY| line: CP2K did not finish")

  # inspect
  supercell = read_one(SUPERCELL)
  if supercell is None:
    steps["inspect"].mark("missing")

  # relax-h
  s = steps["relax-h"]
  hrelaxed = read_one(HRELAXED)
  if hrelaxed is None:
    s.mark("missing")
  else:
    if not hrelaxed.info.get("h_relaxed"):
      s.mark("bad", "no h_relaxed flag: not written by relax-h")
    if not hrelaxed.info.get("h_relax_converged", False):
      s.mark("bad", "relax-h did not converge (h_relax_converged = False)")
    if supercell is not None:
      is_h = np.array(hrelaxed.get_chemical_symbols()) == "H"
      # is_h: (n_atoms,)
      if not same_positions(hrelaxed, supercell, ~is_h):
        s.mark("bad", "atoms, cell or heavy-atom positions differ from supercell.extxyz")
      if mtime(HRELAXED) < mtime(SUPERCELL):
        s.mark("stale", "older than supercell.extxyz")
  if RELAX_CKPT.exists() and mtime(RELAX_CKPT) > mtime(HRELAXED):
    s.mark("partial", "checkpoint newer than the output: a relax-h run is in progress or was interrupted")

  # generate
  s = steps["generate"]
  unlabeled, err = read_frames(UNLABELED) if UNLABELED.exists() else ([], None)
  if not unlabeled:
    s.mark("missing" if not UNLABELED.exists() else "bad", err)
  else:
    ref = next((f for f in unlabeled if f.info.get("config_type") == "reference"), None)
    if ref is None:
      s.mark("bad", "no reference frame")
    elif hrelaxed is not None and not same_positions(ref, hrelaxed):
      if same_positions(ref, supercell):
        s.mark("bad", "generated from the unrelaxed supercell.extxyz, not supercell_hrelaxed.extxyz")
      else:
        s.mark("bad", "reference frame differs from supercell_hrelaxed.extxyz")
    if HRELAXED.exists() and mtime(UNLABELED) < mtime(HRELAXED):
      s.mark("stale", "older than supercell_hrelaxed.extxyz")
  if hrelaxed is not None and unlabeled and supercell is not None:
    sizes = {len(supercell), len(hrelaxed), len(unlabeled[0])}
    if len(sizes) > 1:
      s.mark("bad", f"atom counts disagree across supercell/hrelaxed/unlabeled: {sorted(sizes)}")

  # label
  s = steps["label"]
  label = None
  if not LABELED.exists():
    s.mark("missing")
  else:
    labeled, err = read_frames(LABELED)
    label = label_progress(labeled, unlabeled)
    label["read_error"] = err
    if err:
      s.notes.append(f"read stopped after {len(labeled)} frames ({err}); may be mid-append")
    if label["mismatched"]:
      s.mark("bad", f"{len(label['mismatched'])} labeled frame(s) do not match unlabeled.extxyz "
                    f"(frame_id {label['mismatched']}): relabel them")
    if label["duplicates"]:
      s.notes.append(f"duplicate frame_ids {label['duplicates']} (first copy is used)")
    if label["unknown"]:
      s.mark("bad", f"frame_ids {label['unknown']} are not in unlabeled.extxyz")
    if unlabeled and mtime(LABELED) < mtime(UNLABELED) and not label["mismatched"]:
      s.notes.append("older than unlabeled.extxyz, but every labeled frame matches its geometry")
    if label["n_done"] < label["n_total"]:
      s.mark("partial")

  # staleness passes down the chain
  chain = ["inspect", "relax-h", "generate", "label"]
  for up, down in zip(chain, chain[1:]):
    if steps[up].state in ("stale", "missing") and steps[down].path.exists() and steps[down].state != "stale":
      steps[down].mark("stale", f"upstream step {up} is {steps[up].state}")

  return {"steps": steps, "relax": relax_progress(), "label": label}


def relax_progress():
  """Parse the relax-h progress log.

  Returns
  -------
  dict or None
    "step", "time", "energy", "f_h", "fmax", "restarts", "sec_per_step"; None if
    the log is missing or has no steps.
  """
  if not RELAX_LOG.exists():
    return None
  rows, restarts = [], 0
  for line in RELAX_LOG.read_text().splitlines():
    if line.startswith("# restart"):
      restarts += 1
    elif line and not line.startswith("#"):
      parts = line.split("\t")
      if len(parts) == 5:
        rows.append(parts)
  if not rows:
    return None
  last = rows[-1]
  times = [datetime.fromisoformat(r[1]).timestamp() for r in rows[-6:]]
  sec_per_step = (times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else None
  return {"step": int(last[0]), "time": last[1], "energy": float(last[2]), "f_h": float(last[3]),
          "fmax": float(last[4]), "restarts": restarts, "sec_per_step": sec_per_step}


def label_progress(labeled: list, unlabeled: list) -> dict:
  """Match labeled frames to unlabeled ones by ``frame_id``.

  Parameters
  ----------
  labeled : list of Atoms
    Frames from ``labeled.extxyz``.
  unlabeled : list of Atoms
    Frames from ``unlabeled.extxyz``.

  Returns
  -------
  dict
    Counts per config_type ("done", "total"), "n_done"/"n_total" (matching
    frames only), "mismatched", "duplicates", "unknown" frame_id lists, scan
    angles "phi_done"/"phi_missing", and "label_seconds" (list of float).
  """
  by_id = {int(f.info["frame_id"]): f for f in unlabeled if "frame_id" in f.info}
  seen, duplicates, mismatched, unknown, seconds = {}, [], [], [], []
  for f in labeled:
    fid = int(f.info.get("frame_id", -1))
    if fid in seen:
      duplicates.append(fid)
      continue
    seen[fid] = f
    if "label_seconds" in f.info:
      seconds.append(float(f.info["label_seconds"]))
    if fid not in by_id:
      unknown.append(fid)
    elif not same_positions(f, by_id[fid]):
      mismatched.append(fid)
  good = {fid for fid in seen if fid in by_id and fid not in mismatched}
  total = Counter(f.info.get("config_type", "unknown") for f in by_id.values())
  done = Counter(by_id[fid].info.get("config_type", "unknown") for fid in good)
  scan = {fid: float(f.info["phi_deg"]) for fid, f in by_id.items() if f.info.get("config_type") == "rigid_scan"}
  return {
    "total": dict(total), "done": dict(done),
    "n_total": len(by_id), "n_done": len(good), "n_labeled_unique": len(seen),
    "mismatched": sorted(mismatched), "duplicates": sorted(set(duplicates)), "unknown": sorted(unknown),
    "phi_done": sorted(phi for fid, phi in scan.items() if fid in good),
    "phi_missing": sorted(phi for fid, phi in scan.items() if fid not in good),
    "label_seconds": seconds,
  }


# ---------------------------------------------------------------- jobs and CP2K output

def running_jobs():
  """Processes whose command line mentions cp2k or make_dataset.py.

  Returns
  -------
  list of str or None
    ``"pid etime command"`` lines; None if ``ps`` is not available.
  """
  try:
    out = subprocess.run(["ps", "-axo", "pid=,etime=,command="], capture_output=True, text=True, timeout=10).stdout
  except (FileNotFoundError, subprocess.SubprocessError):
    return None
  return [l.strip() for l in out.splitlines()
          if ("cp2k" in l or "make_dataset.py" in l) and "status.py" not in l]


def cp2k_out_state(path: Path) -> dict:
  """Last SCF line, last energy and end state of a CP2K output file.

  Parameters
  ----------
  path : Path
    CP2K ``.out`` file.

  Returns
  -------
  dict
    "scf" (str or None), "energy_ha" (float or None), "ended" (bool),
    "active" (bool: written within ``ACTIVE_SECONDS`` and not ended).
  """
  scf, energy, ended = None, None, False
  for line in path.read_text(errors="replace").splitlines():
    if SCF_LINE.match(line):
      scf = line.strip()
    elif line.startswith(" ENERGY|"):
      energy = float(line.split()[-1])
    elif "PROGRAM ENDED" in line:
      ended = True
    elif "PROGRAM STARTED" in line:
      ended = False
  active = not ended and time.time() - mtime(path) < ACTIVE_SECONDS
  return {"scf": scf, "energy_ha": energy, "ended": ended, "active": active}


# ---------------------------------------------------------------- models

def model_reports(n_labeled: int) -> list:
  """Benchmark results of every model folder (``models/uma-*``, then ``models/mace-off*``).

  Parameters
  ----------
  n_labeled : int
    Number of unique labeled frames, to detect benchmarks run on fewer frames.

  Returns
  -------
  list of dict
    One per model: "name", "dir", "venv", "tuned" (bool), "reports" (list of
    dicts with "label", "path", "metrics", "stale").
  """
  out = []
  model_dirs = [(mdir, venv) for pattern, venv in MODEL_FAMILIES for mdir in sorted(MODELS.glob(pattern))]
  for mdir, venv in model_dirs:
    if not mdir.is_dir():
      continue
    reports = []
    zero = mdir / "zero_shot" / "results.json"
    if zero.exists():
      reports.append({"label": "zero_shot", "path": zero})
    bench = sorted((mdir / "finetune" / "benchmark").glob("*/results.json"))
    for path in bench:
      reports.append({"label": f"tuned/{path.parent.name}", "path": path})
    split_total = 0
    for r in reports:
      try:
        with open(r["path"]) as fh:
          r["metrics"] = json.load(fh)["metrics"]
      except Exception as err:
        r["metrics"], r["error"] = {}, str(err)
      n = r["metrics"].get("all", {}).get("n_frames", 0)
      r["n_frames"] = n
      if r["label"] in ("tuned/train", "tuned/val", "tuned/test"):
        split_total += n
      r["stale"] = mtime(r["path"]) < mtime(LABELED)
    for r in reports:
      if r["label"] in ("tuned/train", "tuned/val", "tuned/test"):
        r["stale"] = r["stale"] or split_total != n_labeled
      else:
        r["stale"] = r["stale"] or r["n_frames"] != n_labeled
    out.append({"name": mdir.name, "dir": mdir, "venv": venv, "tuned": (mdir / "finetune" / "tuned.pt").exists(),
                "reports": reports})
  return out


def fmt_fit(fit: dict) -> str:
  """``V3 .. V6 .. phi0 ..`` from a fit_barrier dict, or ``-``."""
  if not fit:
    return "-"
  return f"V3 {fit['V3_meV']:6.1f} V6 {fit['V6_meV']:6.1f} meV phi0 {fit['phi0_deg']:5.1f}"


# ---------------------------------------------------------------- next step

def next_step(status: dict, label_active: bool, relax_active: bool) -> str:
  """The single command (or action) for the first incomplete or stale item."""
  steps, label = status["steps"], status["label"]
  if steps["smoke"].state != "ok":
    return "python make_dataset.py smoke"
  if steps["inspect"].state != "ok":
    cifs = sorted(p.name for p in ROOT.glob("*.cif"))
    return f"python make_dataset.py inspect {cifs[0] if len(cifs) == 1 else '<structure.cif>'}"
  if steps["relax-h"].state != "ok":
    if relax_active:
      return "wait: relax-h is running (see relax-h progress above)"
    restart = " --restart" if RELAX_CKPT.exists() else ""
    if HRELAXED.exists() and steps["relax-h"].state == "bad" and mtime(HRELAXED) >= mtime(RELAX_CKPT):
      return (f"relax-h finished but {rel(HRELAXED)} is not usable (see above); check {rel(RELAX_LOG)}, "
              f"delete the relax-h checkpoint files and rerun relax-h without --restart")
    if steps["relax-h"].state == "stale" and RELAX_CKPT.exists() and mtime(RELAX_CKPT) < mtime(SUPERCELL):
      return (f"supercell.extxyz is newer than the relax-h checkpoint: delete {rel(RELAX_CKPT)}, "
              f"{rel(RELAX_LOG)} and the .bfgs.json, then "
              f"python make_dataset.py relax-h {rel(SUPERCELL)} --fmax {RELAX_FMAX}")
    return f"python make_dataset.py relax-h {rel(SUPERCELL)} --fmax {RELAX_FMAX}{restart}"
  if steps["generate"].state != "ok":
    return f"python make_dataset.py generate {rel(HRELAXED)} --rotor 0"
  if label and label["mismatched"]:
    bad = set(label["mismatched"])
    stop = "stop the label run, then " if label_active else ""
    return (f"{stop}drop the mismatched frames so label redoes them:\n"
            f"  python -c \"from ase.io import read, write; "
            f"fr = [a for a in read('{rel(LABELED)}', ':') if a.info['frame_id'] not in {bad}]; "
            f"write('{rel(LABELED)}.new', fr, format='extxyz')\" && mv {rel(LABELED)}.new {rel(LABELED)}")
  if steps["label"].state != "ok":
    if label_active:
      return "wait: label is running (see label progress above)"
    return f"python make_dataset.py label {rel(UNLABELED)}"
  for model in status.get("models", []):
    zero = next((r for r in model["reports"] if r["label"] == "zero_shot"), None)
    if zero is None or zero["stale"]:
      return f"python {rel(model['dir'])}/zero_shot.py   (in {model['venv']})"
  for model in status.get("models", []):
    if model["tuned"] and any(r["stale"] for r in model["reports"] if r["label"].startswith("tuned/")):
      return f"python {rel(model['dir'])}/benchmark_tuned.py   (in {model['venv']})"
  return "nothing: every step is complete and up to date"


# ---------------------------------------------------------------- report

def report() -> str:
  """Build the full status report as text."""
  now = time.time()
  lines = [f"methyl pipeline status  {datetime.now():%Y-%m-%d %H:%M:%S}  ({ROOT})", ""]
  status = check_pipeline()
  steps = status["steps"]

  lines.append("Pipeline")
  for step in steps.values():
    lines.append(step.line())
    lines += [f"      {n}" for n in step.notes]
  ref = read_one(HRELAXED) or read_one(SUPERCELL)
  if ref is not None:
    lengths = " x ".join(f"{x:.2f}" for x in ref.cell.lengths())
    lines.append(f"  cell {lengths} A, {len(ref)} atoms")

  # CP2K output files: also used to decide whether a job is running (ps cannot see Mac processes from Docker)
  recent = sorted((p for pat in ("*.out", "*.log") for p in ROOT.glob(pat) if now - mtime(p) < RECENT_SECONDS),
                  key=mtime, reverse=True)
  cp2k = {p.stem: cp2k_out_state(p) for p in ROOT.glob("*.out")}
  jobs = running_jobs()
  relax_active = cp2k.get("relax_h", {}).get("active", False) or any("relax-h" in j for j in jobs or [])
  label_active = cp2k.get("label", {}).get("active", False) or any(" label " in j for j in jobs or [])

  relax = status["relax"]
  if relax:
    lines += ["", "relax-h progress  " + rel(RELAX_LOG)]
    conv = OK if relax["f_h"] < relax["fmax"] else WARN
    rate = f", {relax['sec_per_step'] / 60:.1f} min/step" if relax["sec_per_step"] else ""
    lines.append(f"  step {relax['step']} at {relax['time']}: max H force {relax['f_h']:.4f} eV/A "
                 f"(fmax {relax['fmax']:g}) {conv}, E {relax['energy']:.5f} eV{rate}")
    lines.append(f"  restarts {relax['restarts']}" + ("; CP2K active now" if relax_active else ""))

  label = status["label"]
  if label:
    lines += ["", "Label progress  " + rel(LABELED)]
    lines.append(f"  {label['n_done']}/{label['n_total']} frames done"
                 + (f" ({label['n_labeled_unique']} in file)" if label["n_labeled_unique"] != label["n_done"] else ""))
    for ctype, n in sorted(label["total"].items()):
      lines.append(f"    {ctype:12s} {label['done'].get(ctype, 0):3d}/{n}")
    if label["phi_done"] or label["phi_missing"]:
      lines.append("  rigid_scan done:    " + (" ".join(f"{p:g}" for p in label["phi_done"]) or "-"))
      lines.append("  rigid_scan missing: " + (" ".join(f"{p:g}" for p in label["phi_missing"]) or "-"))
      if len(label["phi_done"]) < MIN_SCAN_ANGLES:
        lines.append(f"  {WARN} fewer than {MIN_SCAN_ANGLES} scan angles: no V3/V6 fit yet")
    remaining = label["n_total"] - label["n_done"]
    if label["label_seconds"]:
      mean = float(np.mean(label["label_seconds"]))
      lines.append(f"  {mean / 60:.1f} min/frame (mean of {len(label['label_seconds'])}); "
                   f"ETA {fmt_age(mean * remaining)} for {remaining} frames")
    elif remaining:
      lines.append("  no label_seconds recorded yet (frames from an older make_dataset.py): no ETA")
    label_out = ROOT / "label.out"
    if remaining and label_out.exists() and not label_active:
      st = cp2k["label"]
      why = "finished its last frame" if st["ended"] else "stopped mid-calculation: killed or stalled"
      lines.append(f"  {WARN} label is not running: label.out last written {fmt_age(now - mtime(label_out))} ago, {why}")

  lines += ["", "Running jobs"]
  if jobs is None:
    lines.append("  ps not available (e.g. in the Docker container, which cannot see Mac processes); "
                 "judging from CP2K .out activity instead")
  elif not jobs:
    lines.append("  none")
  else:
    lines += [f"  {j}" for j in jobs]
  for name, st in sorted(cp2k.items()):
    if st["active"]:
      lines.append(f"  {name}.out: CP2K writing (updated {fmt_age(now - mtime(ROOT / (name + '.out')))} ago)")

  lines += ["", f"Recent files (last {fmt_age(RECENT_SECONDS)})"]
  if not recent:
    lines.append("  none")
  for path in recent:
    age = fmt_age(now - mtime(path))
    if path.suffix == ".out":
      st = cp2k[path.stem]
      state = "ended" if st["ended"] else ("running" if st["active"] else "NOT ENDED, no longer written: killed or stalled")
      lines.append(f"  {path.name:16s} {age:>6s} ago  {state}")
      if st["energy_ha"] is not None:
        lines.append(f"      last ENERGY| {st['energy_ha']:.8f} Ha")
      if st["scf"] and not st["ended"]:
        lines.append(f"      last SCF: {st['scf']}")
    else:
      lines.append(f"  {path.name:16s} {age:>6s} ago")

  n_labeled = label["n_labeled_unique"] if label else 0
  status["models"] = model_reports(n_labeled)
  lines += ["", "Models"]
  if not status["models"]:
    lines.append("  none")
  for model in status["models"]:
    tuned = "tuned.pt " + (OK if model["tuned"] else "-")
    lines.append(f"  {model['name']:12s} {tuned}   ({model['venv']})")
    if not model["reports"]:
      lines.append("      no results.json yet")
    for r in model["reports"]:
      m = r["metrics"]
      a = m.get("all", {})
      tag = f"  [STALE vs {LABELED.name}]" if r["stale"] else ""
      if "error" in r:
        lines.append(f"      {r['label']:14s} unreadable: {r['error']}")
        continue
      lines.append(f"      {r['label']:14s} n={a.get('n_frames', 0):3d}  F MAE {a.get('force_mae_eV_A', float('nan')):.4f}"
                   f"  F_H MAE {a.get('force_mae_H_eV_A', float('nan')):.4f} eV/A{tag}")
      scan = m.get("scan")
      if scan:
        lines.append(f"      {'':14s} DFT   {fmt_fit(scan.get('fit_dft'))}")
        lines.append(f"      {'':14s} model {fmt_fit(scan.get('fit_model'))}")

  lines += ["", "Next step", "  " + next_step(status, label_active, relax_active)]
  return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
  """Command-line parser."""
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--watch", type=float, metavar="N", help="redraw every N seconds until Ctrl-C")
  return parser


def main() -> None:
  """Print the report once, or redraw it every ``--watch`` seconds."""
  args = build_parser().parse_args()
  if not args.watch:
    print(report())
    return
  try:
    while True:
      text = report()
      sys.stdout.write("\033[2J\033[H" + text + "\n")
      sys.stdout.flush()
      time.sleep(args.watch)
  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  main()
