"""Methyl tunneling splittings for every benchmarked model.

Finds each ``results.json`` under ``models/``, takes the rigid-scan energies it
stores (DFT and model), fits V(phi) as a threefold Fourier series, solves the 1D
hindered-rotor Schroedinger equation and compiles the A-E ground-state splitting
for CH3 and CD3. Each row is labeled ``zero-shot``, ``fine-tuned`` or
``DFT reference``.

Usage (from the code folder, no DFT or model needed, runs in under a second):
  python tunneling/splitting.py                 # scan models/, write tunneling/splittings.{csv,json}
  python tunneling/splitting.py --self-test     # check the solver against known limits
  python tunneling/splitting.py --root models/uma-s-1p1 --harmonics 4

Physics: H = -B d2/dphi2 + V(phi), V(phi) = sum_k c_k cos(3k phi) + s_k sin(3k phi).
In the free-rotor basis exp(i m phi) the potential only couples m to m +/- 3k, so
H splits into an A block (m = 0 mod 3) and an E block (m = 1 mod 3; m = 2 mod 3 is
degenerate with it). The tunneling splitting is E_E,0 - E_A,0.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

B_CH3_MEV = 0.655
"""Rotational constant hbar^2 / 2I of CH3 in meV (project default)."""

MASS_RATIO_H_D = 1.00782503 / 2.01410178
"""m_H / m_D; B(CD3) = B(CH3) * MASS_RATIO_H_D for the same geometry."""

MHZ_PER_UEV = 241.79893
"""Frequency of 1 micro-eV in MHz."""

PERIOD_DEG = 120.0
DFT_LABEL = "CP2K PBE-D3(BJ)"
COLUMNS = [
  "model", "mode", "eval", "isotope", "B_meV", "V3_meV", "V6_meV", "phi0_deg", "fit_rmse_meV",
  "barrier_meV", "splitting_ueV", "splitting_MHz", "splitting_v3v6_ueV", "torsion_meV",
  "n1_splitting_ueV", "ratio_to_dft", "source",
]


# ---------------------------------------------------------------- potential

def unique_scan(phi_deg: np.ndarray, energy: np.ndarray) -> tuple:
  """Fold scan angles into one 120 degree period and drop repeated angles.

  A 0-120 degree scan holds the same geometry twice (0 and 120); keeping both
  would double-weight that point in the fit.

  Parameters
  ----------
  phi_deg : np.ndarray
    Scan angles in degrees. Shape: (n_scan,).
  energy : np.ndarray
    Energies at those angles, any unit and offset. Shape: (n_scan,).

  Returns
  -------
  tuple of (np.ndarray, np.ndarray)
    Sorted unique angles in [0, 120) and their energies. Shape: (n_unique,) each.
  """
  folded = np.round(np.mod(np.asarray(phi_deg, dtype=float), PERIOD_DEG), 6)
  # folded: (n_scan,)
  angles, first = np.unique(folded, return_index=True)
  # angles, first: (n_unique,)
  return angles, np.asarray(energy, dtype=float)[first]


def fit_fourier(phi_deg: np.ndarray, energy: np.ndarray, n_harmonics: int = 5) -> tuple:
  """Least-squares threefold Fourier series through scan energies.

  V(phi) = sum_{k=0..K} cos_coef[k] cos(3k phi) + sin_coef[k] sin(3k phi). K is
  reduced if there are too few angles to determine 2K + 1 coefficients.

  Parameters
  ----------
  phi_deg : np.ndarray
    Unique angles within one period, degrees. Shape: (n_unique,).
  energy : np.ndarray
    Energies at those angles. Shape: (n_unique,).
  n_harmonics : int, optional
    Highest harmonic K (the term cos 3K phi). Defaults to 5.

  Returns
  -------
  tuple of (np.ndarray, np.ndarray)
    (cos_coef, sin_coef) in the units of ``energy``; sin_coef[0] is 0.
    Shape: (K + 1,) each.
  """
  n_harm = max(1, min(n_harmonics, (len(phi_deg) - 1) // 2))
  phi = np.radians(phi_deg)
  # phi: (n_unique,)
  columns = [np.ones_like(phi)]
  for k in range(1, n_harm + 1):
    columns += [np.cos(3 * k * phi), np.sin(3 * k * phi)]
  design = np.column_stack(columns)
  # design: (n_unique, 2K + 1)
  coef, *_ = np.linalg.lstsq(design, energy, rcond=None)
  # coef: (2K + 1,)
  cos_coef = np.concatenate([coef[:1], coef[1::2]])
  sin_coef = np.concatenate([[0.0], coef[2::2]])
  # cos_coef, sin_coef: (K + 1,)
  return cos_coef, sin_coef


def coefficients_from_v3v6(v3: float, v6: float, phi0_deg: float) -> tuple:
  """Fourier coefficients of V3/2 (1 - cos 3(phi - phi0)) + V6/2 (1 - cos 6(phi - phi0)).

  Parameters
  ----------
  v3 : float
    Threefold barrier term.
  v6 : float
    Signed sixfold term.
  phi0_deg : float
    Phase offset of the minimum in degrees.

  Returns
  -------
  tuple of (np.ndarray, np.ndarray)
    (cos_coef, sin_coef) in the units of ``v3``. Shape: (3,) each.
  """
  phi0 = np.radians(phi0_deg)
  cos_coef = np.array([(v3 + v6) / 2, -v3 / 2 * np.cos(3 * phi0), -v6 / 2 * np.cos(6 * phi0)])
  sin_coef = np.array([0.0, -v3 / 2 * np.sin(3 * phi0), -v6 / 2 * np.sin(6 * phi0)])
  # cos_coef, sin_coef: (3,)
  return cos_coef, sin_coef


def barrier_height(cos_coef: np.ndarray, sin_coef: np.ndarray) -> float:
  """Maximum minus minimum of the Fourier-series potential over one period.

  Parameters
  ----------
  cos_coef : np.ndarray
    Cosine coefficients. Shape: (K + 1,).
  sin_coef : np.ndarray
    Sine coefficients. Shape: (K + 1,).

  Returns
  -------
  float
    Barrier height in the units of the coefficients.
  """
  phi = np.radians(np.linspace(0.0, PERIOD_DEG, 1201))
  # phi: (n_grid,)
  k = np.arange(len(cos_coef))
  # k: (K + 1,)
  values = np.cos(3 * np.outer(phi, k)) @ cos_coef + np.sin(3 * np.outer(phi, k)) @ sin_coef
  # values: (n_grid,)
  return float(values.max() - values.min())


# ---------------------------------------------------------------- rotor solver

def rotor_levels(cos_coef: np.ndarray, sin_coef: np.ndarray, b_mev: float, m_max: int = 60,
                 n_levels: int = 4) -> dict:
  """Lowest hindered-rotor levels of each symmetry by exact diagonalisation.

  Parameters
  ----------
  cos_coef : np.ndarray
    Cosine coefficients of V(phi) in meV. Shape: (K + 1,).
  sin_coef : np.ndarray
    Sine coefficients of V(phi) in meV. Shape: (K + 1,).
  b_mev : float
    Rotational constant hbar^2 / 2I in meV.
  m_max : int, optional
    Basis cutoff, |m| <= about m_max. Defaults to 60 (converged to ~1e-9 micro-eV
    for barriers up to a few hundred meV).
  n_levels : int, optional
    Number of levels returned per symmetry. Defaults to 4.

  Returns
  -------
  dict
    "A" and "E": ascending level energies in meV. Shape: (n_levels,) each.
    Each E level is doubly degenerate.
  """
  n_side = m_max // 3
  offsets = np.arange(-n_side, n_side + 1)
  # offsets: (n_basis,)
  levels = {}
  for name, residue in (("A", 0), ("E", 1)):
    m = residue + 3 * offsets
    # m: (n_basis,), free-rotor quantum numbers of this symmetry block
    ham = np.diag(b_mev * m.astype(float) ** 2 + cos_coef[0]).astype(complex)
    # ham: (n_basis, n_basis)
    for k in range(1, len(cos_coef)):
      # <m|V|m'> with m - m' = 3k is the exp(i 3k phi) coefficient (c_k - i s_k) / 2
      upper = np.full(len(m) - k, (cos_coef[k] + 1j * sin_coef[k]) / 2)
      # upper: (n_basis - k,)
      ham += np.diag(upper, k) + np.diag(upper.conj(), -k)
    levels[name] = np.linalg.eigvalsh(ham)[:n_levels]
  return levels


def tunneling_observables(cos_coef: np.ndarray, sin_coef: np.ndarray, b_mev: float) -> dict:
  """Tunneling splitting and torsional energy for one potential and one isotope.

  Parameters
  ----------
  cos_coef : np.ndarray
    Cosine coefficients of V(phi) in meV. Shape: (K + 1,).
  sin_coef : np.ndarray
    Sine coefficients of V(phi) in meV. Shape: (K + 1,).
  b_mev : float
    Rotational constant in meV.

  Returns
  -------
  dict
    "splitting_ueV" (E_E,0 - E_A,0), "splitting_MHz", "torsion_meV" (0 -> 1
    within the A levels, E_A,1 - E_A,0, the convention of the project reference
    table) and "n1_splitting_ueV" (E_A,1 - E_E,1; positive because E lies below
    A at n = 1).
  """
  levels = rotor_levels(cos_coef, sin_coef, b_mev)
  a, e = levels["A"], levels["E"]
  # a, e: (n_levels,)
  splitting = (e[0] - a[0]) * 1e3
  return {
    "splitting_ueV": float(splitting),
    "splitting_MHz": float(splitting * MHZ_PER_UEV),
    "torsion_meV": float(a[1] - a[0]),
    "n1_splitting_ueV": float((a[1] - e[1]) * 1e3),
  }


# ---------------------------------------------------------------- results files

def find_results(root: Path) -> list:
  """All benchmark result files below a folder.

  Parameters
  ----------
  root : Path
    Folder to search (normally ``models/``).

  Returns
  -------
  list of Path
    Sorted paths of every ``results.json`` found.
  """
  return sorted(root.rglob("results.json"))


def describe_run(meta: dict, path: Path) -> dict:
  """Model name, zero-shot / fine-tuned label and evaluation set of one result file.

  Parameters
  ----------
  meta : dict
    The "meta" entry of results.json.
  path : Path
    Location of that file (fallback when meta lacks a field).

  Returns
  -------
  dict
    "model" (base model name), "mode" ("zero-shot" or "fine-tuned") and "eval"
    (which frames were scored: "all", a split name, or the folder name).
  """
  tuned = bool(meta.get("tuned", "finetune" in path.parts))
  model = meta.get("base_model") or str(meta.get("model", path.parts[-3])).replace(" fine-tuned", "")
  if tuned:
    evaluated = meta.get("eval_split") or path.parent.name
  else:
    evaluated = "all"
  return {"model": model, "mode": "fine-tuned" if tuned else "zero-shot", "eval": evaluated}


def analyse_curve(phi_deg: np.ndarray, energy_mev: np.ndarray, fit: dict, b_values: dict,
                  n_harmonics: int) -> list:
  """Splittings of one scan curve for every isotope.

  Parameters
  ----------
  phi_deg : np.ndarray
    Scan angles in degrees. Shape: (n_scan,).
  energy_mev : np.ndarray
    Scan energies in meV, any offset. Shape: (n_scan,).
  fit : dict
    Saved V3/V6/phi0 fit ("V3_meV", "V6_meV", "phi0_deg", "rmse_meV"); may be empty.
  b_values : dict
    Isotope name -> rotational constant in meV.
  n_harmonics : int
    Highest harmonic of the Fourier fit.

  Returns
  -------
  list of dict
    One partial row per isotope (no model/mode/source fields yet).
  """
  angles, energies = unique_scan(phi_deg, energy_mev)
  # angles, energies: (n_unique,)
  cos_coef, sin_coef = fit_fourier(angles, energies, n_harmonics)
  # cos_coef, sin_coef: (K + 1,)
  rows = []
  for isotope, b_mev in b_values.items():
    row = {"isotope": isotope, "B_meV": b_mev, "V3_meV": fit.get("V3_meV"), "V6_meV": fit.get("V6_meV"),
           "phi0_deg": fit.get("phi0_deg"), "fit_rmse_meV": fit.get("rmse_meV"),
           "barrier_meV": barrier_height(cos_coef, sin_coef)}
    row.update(tunneling_observables(cos_coef, sin_coef, b_mev))
    row["splitting_v3v6_ueV"] = None
    if fit:
      simple = coefficients_from_v3v6(fit["V3_meV"], fit["V6_meV"], fit["phi0_deg"])
      row["splitting_v3v6_ueV"] = tunneling_observables(*simple, b_mev)["splitting_ueV"]
    rows.append(row)
  return rows


def collect(root: Path, base: Path, b_values: dict, n_harmonics: int) -> tuple:
  """Compile splittings for every result file below ``root``.

  Parameters
  ----------
  root : Path
    Folder searched for results.json files.
  base : Path
    Folder that "source" paths are written relative to.
  b_values : dict
    Isotope name -> rotational constant in meV.
  n_harmonics : int
    Highest harmonic of the Fourier fit.

  Returns
  -------
  tuple of (list of dict, list of str)
    (rows with the keys in ``COLUMNS``, DFT reference rows first; messages about
    files that were skipped).
  """
  dft_rows, model_rows, skipped, seen_dft = [], [], [], set()
  for path in find_results(root):
    with open(path) as fh:
      data = json.load(fh)
    scan = data.get("metrics", {}).get("scan")
    source = str(path.relative_to(base)) if path.is_relative_to(base) else str(path)
    if not scan or len(scan.get("phi_deg", [])) < 5:
      skipped.append(f"{source}: no rigid-scan frames in this evaluation set")
      continue
    meta = data.get("meta", {})
    run = describe_run(meta, path)
    phi = np.array(scan["phi_deg"])
    # phi: (n_scan,)
    dft = analyse_curve(phi, np.array(scan["dE_dft_meV"]), scan.get("fit_dft") or {}, b_values, n_harmonics)
    model = analyse_curve(phi, np.array(scan["dE_model_meV"]), scan.get("fit_model") or {}, b_values, n_harmonics)
    for dft_row, model_row in zip(dft, model):
      model_row["ratio_to_dft"] = model_row["splitting_ueV"] / dft_row["splitting_ueV"]
      model_rows.append({**run, **model_row, "source": source})
    dft_key = (meta.get("labeled"), tuple(np.round(scan["dE_dft_meV"], 6)))
    if dft_key not in seen_dft:
      seen_dft.add(dft_key)
      labeled = meta.get("labeled") or source
      for dft_row in dft:
        dft_rows.append({"model": DFT_LABEL, "mode": "DFT reference", "eval": "scan", **dft_row,
                         "ratio_to_dft": 1.0, "source": labeled})
  order = {"zero-shot": 0, "fine-tuned": 1}
  model_rows.sort(key=lambda r: (r["model"], order[r["mode"]], r["eval"], r["isotope"] != "CH3"))
  return dft_rows + model_rows, skipped


# ---------------------------------------------------------------- reporting

def format_value(value) -> str:
  """Short text form of one table cell.

  Parameters
  ----------
  value : float, str or None
    Cell content.

  Returns
  -------
  str
    Four significant figures for numbers, "-" for None, text unchanged.
  """
  if value is None:
    return "-"
  if isinstance(value, str):
    return value
  return f"{value:.4g}"


def print_table(rows: list) -> None:
  """Print the compiled splittings as an aligned text table.

  Parameters
  ----------
  rows : list of dict
    Output of ``collect``.

  Returns
  -------
  None
  """
  shown = [c for c in COLUMNS if c not in ("source", "B_meV", "splitting_MHz", "n1_splitting_ueV")]
  cells = [shown] + [[format_value(row[c]) for c in shown] for row in rows]
  widths = [max(len(line[i]) for line in cells) for i in range(len(shown))]
  for line in cells:
    print("  ".join(text.ljust(width) for text, width in zip(line, widths)))


def write_outputs(rows: list, out_dir: Path, settings: dict) -> None:
  """Write splittings.csv and splittings.json.

  Parameters
  ----------
  rows : list of dict
    Output of ``collect``.
  out_dir : Path
    Output folder, created if missing.
  settings : dict
    Run settings (rotational constants, harmonics, search root) stored in the JSON.

  Returns
  -------
  None
  """
  out_dir.mkdir(parents=True, exist_ok=True)
  with open(out_dir / "splittings.csv", "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
  with open(out_dir / "splittings.json", "w") as fh:
    json.dump({"settings": settings, "rows": rows}, fh, indent=2)


def self_test() -> None:
  """Check the solver against the free-rotor limit and the project reference table.

  Returns
  -------
  None
  """
  free = tunneling_observables(np.array([0.0, 0.0]), np.array([0.0, 0.0]), B_CH3_MEV)
  assert abs(free["splitting_ueV"] - B_CH3_MEV * 1e3) < 1e-6, free
  # reference values: single rotor, CH3, B = 0.655 meV (project context, section 4)
  for v3, split_ref, torsion_ref in ((20, 38.5, 9.54), (40, 3.22, 13.75), (60, 0.428, 17.19), (80, 0.0746, 20.12)):
    coef = coefficients_from_v3v6(float(v3), 0.0, 0.0)
    got = tunneling_observables(*coef, B_CH3_MEV)
    assert abs(got["splitting_ueV"] / split_ref - 1) < 5e-3, (v3, got)
    assert abs(got["torsion_meV"] - torsion_ref) < 0.01, (v3, got)
    shifted = tunneling_observables(*coefficients_from_v3v6(float(v3), 0.0, 17.0), B_CH3_MEV)
    assert abs(shifted["splitting_ueV"] / got["splitting_ueV"] - 1) < 1e-6, (v3, shifted)
  phi = np.arange(0.0, 120.0, 10.0)
  # phi: (12,)
  energy = 20.0 * (1 - np.cos(3 * np.radians(phi - 22.0))) + 1.5 * (1 - np.cos(6 * np.radians(phi - 22.0)))
  fitted = tunneling_observables(*fit_fourier(phi, energy), B_CH3_MEV)
  exact = tunneling_observables(*coefficients_from_v3v6(40.0, 3.0, 22.0), B_CH3_MEV)
  assert abs(fitted["splitting_ueV"] / exact["splitting_ueV"] - 1) < 1e-6, (fitted, exact)
  print("self-test passed: free-rotor limit, reference table (V3 = 20-80 meV), phase invariance, Fourier fit")


def main() -> None:
  """Parse arguments, compile splittings for all result files and write the tables.

  Returns
  -------
  None
  """
  code_dir = Path(__file__).resolve().parent.parent
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  parser.add_argument("--root", type=Path, default=code_dir / "models", help="folder searched for results.json")
  parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent, help="where the tables go")
  parser.add_argument("--b-ch3", type=float, default=B_CH3_MEV, help="CH3 rotational constant in meV")
  parser.add_argument("--harmonics", type=int, default=5, help="highest harmonic K of the fit (cos 3K phi)")
  parser.add_argument("--self-test", action="store_true", help="check the solver and exit")
  args = parser.parse_args()
  if args.self_test:
    self_test()
    return

  b_values = {"CH3": args.b_ch3, "CD3": args.b_ch3 * MASS_RATIO_H_D}
  rows, skipped = collect(args.root.resolve(), code_dir, b_values, args.harmonics)
  for message in skipped:
    print(f"skipped {message}")
  if not rows:
    raise SystemExit(f"No results.json with a rigid scan found under {args.root}.")
  print_table(rows)
  settings = {"root": str(args.root), "B_meV": b_values, "harmonics": args.harmonics}
  write_outputs(rows, args.out_dir, settings)
  n_runs = len({row["source"] for row in rows if row["mode"] != "DFT reference"})
  print(f"\n{n_runs} model run(s); wrote {args.out_dir / 'splittings.csv'} and splittings.json")


if __name__ == "__main__":
  main()
