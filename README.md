# Seismic Column Optimiser — Circular RC Columns on Type II Shafts (Caltrans SDC 2.1 / AASHTO SGS, ESA)

Pure-Python tool with a Streamlit GUI that analyses and optimises **circular
reinforced-concrete columns supported on Type II (enlarged) shafts** for seismic
checks per **Caltrans SDC 2.1 (Jan 2025)** or the **AASHTO Guide Specifications
for LRFD Seismic Bridge Design (3rd Ed.)**, using the **Equivalent Static
Analysis (ESA)** method. It runs a fibre-based **moment-curvature** analysis with
**Mander** confined concrete, idealises to the **Caltrans bilinear** (φy, Mp, φu),
and evaluates displacement demand and capacity plus the full suite of SDC checks
— for a whole **batch** of columns at once.

Code-specific provisions (concrete shear model, minimum transverse reinforcement,
expected-strength floor, Type II shaft oversize and capacity protection,
short-period magnification, detailing) all switch consistently with the selected
code; see [seismic_column/provisions.py](seismic_column/provisions.py).

## Features

- Mander confined-concrete model (circular spiral/hoop) and ASTM A706 steel
  (SDC expected properties, reduced ultimate tensile strain for φu).
- Fibre moment-curvature with equal-area bilinear idealisation.
- **Type II shaft** modelling: plastic hinge held in the **column at top of
  shaft**; two-segment equivalent cantilever to a **point of fixity** at
  `multiplier × shaft diameter` (default **3×** upper-bound and **6×**
  lower-bound stiffness, run as an envelope).
- Cracked stiffness: column `Ieff = Mp/φy`, shaft `Ieff` from its own M-φ, plus
  gross `Ig` and `Ieff/Ig` ratios.
- SDC checks: displacement capacity vs demand, displacement-ductility demand,
  min/max longitudinal reinforcement, minimum transverse reinforcement
  (Caltrans **Table 5.3.8.2-1** lookup, or AASHTO ρs ≥ 0.005), shear, axial-load
  ratio (`ρdl ≤ 0.15`), P-Δ, minimum lateral strength, detailing (tie/bar
  spacing), and **shaft capacity protection** (flexure & shear against the column
  overstrength moment `Mo = 1.2 Mp`).
- **Code-selectable shear model** — Caltrans SDC 2.0 §3.6.2 (psi form,
  `F2 ≤ 1.5`, `vc ≤ 4√f'c`) or AASHTO SGS §8.6.2 (ksi form,
  `vc = 0.032·α'·(1 + Pc/2Ag)·√f'c` capped at `min(0.11√f'c, 0.047α'√f'c)`,
  with `fs = ρs·fyh ≤ 0.35 ksi` and `vc = 0` under net tension). **SDC D** is
  assumed, so `μΔ` in `α'` is the computed displacement-ductility demand.
- Greedy, **priority-ordered optimiser** (default: longitudinal → confinement →
  diameter → f'c) with user-selectable fixed/variable parameters.
- **Batch tabular** workflow (editable table + CSV/Excel import/export), results
  grid, per-column drill-down with M-φ and spectrum plots, and Markdown reports.

## Units

US customary throughout: **kip, in, ksi**; g = 386.088 in/s². Table inputs use
`Hcol_ft` (feet) and diameters/spacings in inches; weights/loads in kips.

## Install & run

```powershell
uv venv --python 3.14
uv pip install -r requirements.txt pytest
.venv\Scripts\python.exe -m streamlit run app.py
```

The system `python` is not on PATH on this machine, so call the venv
interpreter explicitly (`.venv\Scripts\python.exe`) rather than `python`.

Programmatic use:

```python
from seismic_column.io_schema import default_dataframe, GlobalConfig
from seismic_column.batch import run_batch
summary, results = run_batch(default_dataframe(3), GlobalConfig())
```

Run the tests:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

## Key assumptions & modelling choices

- **Single-column bent** (cantilever): lateral load at column top, plastic hinge
  in the column at the top of shaft.
- **Δy** uses the elastic two-segment cantilever to the point of fixity; **Δp =
  θp·(Hcol − Lp/2)** with `θp = Lp·(φu − φy)`; **Δc = Δy + Δp**.
- **Plastic hinge length**: `Lp = 0.08·Hcol + 0.15·fye·dbl ≥ 0.3·fye·dbl`.
- **Ultimate confined strain**: `εcu = 0.004 + 1.4·ρs·fyh·εsu / f'cc`.
- **Design spectrum**: AASHTO/Caltrans two-parameter form (`Sds`, `Sd1`);
  displacement demand from the equal-displacement rule at the effective
  (cracked) period.
- **Shaft flexural demand basis** is configurable: `interface` (default — the
  column overstrength moment `Mo` at the top of shaft, the standard SDC
  capacity-protection demand) or `fixity` (Mo amplified linearly to the assumed
  point of fixity; conservative, no soil model needed).

> These simplified fixity/hinge assumptions are appropriate for preliminary
> design. Confirm against a soil-structure (e.g. LPILE) model where required.

## Validating against CSiBridge (optional)

The moment-curvature engine is the piece worth cross-checking:

1. In CSiBridge **Section Designer**, build the same circular section (diameter,
   cover, longitudinal bars, spiral) with a **Caltrans** section and matching
   Mander/steel material definitions.
2. Run the **Moment-Curvature** tool at the same constant axial load.
3. Compare the idealised **φy**, **Mp** and **φu** with this tool's report
   (`Moment-curvature (column)` section). Agreement within a few percent is
   expected; differences usually trace to fibre count, cover-spalling treatment,
   or the ultimate-strain limit state.

## Package layout

```
seismic_column/
  materials.py         Mander concrete + A706 steel + bar catalogue
  section.py           circular section geometry + fibre discretisation
  moment_curvature.py  fibre M-φ solver + Caltrans bilinear idealisation
  geometry.py          Type II two-segment equivalent cantilever
  demand.py            design spectrum + ESA displacement demand
  sdc_capacity.py      Lp, Δ-capacity, all SDC checks, shaft capacity protection
  optimizer.py         greedy priority-ordered design search
  io_schema.py         batch table schema + CSV/Excel I/O + validation
  batch.py             batch runner + summary grid
  report.py            per-column Markdown report
app.py                 Streamlit GUI
tests/                 pytest suite
```
