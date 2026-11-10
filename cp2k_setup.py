"""CP2K calculator factory: PBE-D3(BJ), GPW, Gamma point.

PBE-D3 with Becke-Johnson damping is chosen to match the level of theory of
the OMC25 training data (VASP PBE-D3(BJ), per Gharakhanyan et al., arXiv
2508.02651), so reference scans are comparable with OMC25/UMA predictions.
Two-body only (no C9 three-body term), which is the VASP IVDW = 12 default.
The basis set (GTH/MOLOPT vs VASP PAW plane waves) still differs, so absolute
energies are not interchangeable with OMC25 labels.
"""

from ase.calculators.cp2k import CP2K
from ase.units import Rydberg

# Extra CP2K input merged into the input ASE generates. Adds Grimme D3(BJ)
# dispersion. CP2K finds dftd3.dat in $CP2K_DATA_DIR.
# Previously TYPE DFTD3 (zero damping); labels made with that are not
# compatible with labels made with this file.
D3_INP = """
&FORCE_EVAL
  &DFT
    &SCF
      EPS_SCF 1.0E-7
    &END SCF
    &XC
      &VDW_POTENTIAL
        POTENTIAL_TYPE PAIR_POTENTIAL
        &PAIR_POTENTIAL
          TYPE DFTD3(BJ)
          PARAMETER_FILE_NAME dftd3.dat
          REFERENCE_FUNCTIONAL PBE
          R_CUTOFF 15.0
        &END PAIR_POTENTIAL
      &END VDW_POTENTIAL
    &END XC
  &END DFT
&END FORCE_EVAL
"""


def make_cp2k_calc(
  label: str = "cp2k",
  cutoff_ry: float = 500.0,
  command: str = None,
  max_scf: int = 100,
) -> CP2K:
  """Create a CP2K calculator for periodic PBE-D3(BJ) single points and relaxations.

  Parameters
  ----------
  label : str, optional
    Prefix for CP2K's working files. Defaults to "cp2k".
  cutoff_ry : float, optional
    Plane-wave density cutoff in Rydberg. Converge this on one structure
    (energy differences between two rotor angles, not total energies).
    Defaults to 500.
  command : str, optional
    Shell command that starts CP2K in shell mode, e.g.
    "env OMP_NUM_THREADS=1 mpirun -np 8 cp2k.psmp -s". Defaults to the
    $ASE_CP2K_COMMAND environment variable, then "cp2k.psmp -s".
  max_scf : int, optional
    Maximum SCF iterations per outer loop. Defaults to 100.

  Returns
  -------
  CP2K
    Calculator; use it in a ``with`` block so the CP2K process is closed.
  """
  kwargs = dict(
    label=label,
    xc="PBE",
    basis_set="DZVP-MOLOPT-SR-GTH",
    pseudo_potential="GTH-PBE",
    cutoff=cutoff_ry * Rydberg,
    max_scf=max_scf,
    stress_tensor=True,
    inp=D3_INP,
    set_pos_file=True,
  )
  if command is not None:
    kwargs["command"] = command
  return CP2K(**kwargs)
