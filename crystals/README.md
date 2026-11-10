# crystals/

Experimental structures for the `methyl_benchmark.xlsx` Crystals sheet, downloaded from the
Crystallography Open Database (COD, https://www.crystallography.net/cod/) on 2026-10-20.

- `<Crystal_ID>_COD<id>.cif`: one selected structure per crystal (workbook rule 5).
- `selection.csv`: COD id, structure T and P, space group vs. the Prager space group, R, radiation,
  DOI, a quality flag and a list of alternative COD entries.
- `not_in_cod.csv`: in-scope crystals with no usable COD entry; these need the CSD/ICSD or the
  original paper.

Selection rule: the correct compound and phase (checked by name or title, because many formula hits
are isomers), then the lowest temperature at ambient pressure, then the lowest R. Tunnelling spectra
are measured at about 2–10 K, so low-T and neutron (deuterated) structures come first.

Quality flags:
- `good`: low-T, ambient-P, ordered, with H positions.
- `caveat`: usable, but see the note (high-P only, T not recorded, disordered methyls, phase unclear).
- `weak`: last resort (high-P only and far from ambient, or no H atoms).

Benchmark crystals must not be used as training data (workbook rule 3).

`_cod_search/`: `scripts/fetch.py` holds the curated pick list with per-crystal notes; run it to re-download
every CIF and rebuild both CSVs. `scripts/syn2.py` plus `data/syn2.{json,txt}` are the synonym name searches,
filtered to exact formulas. The first-pass formula-search dumps were not kept.
