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
- **Balanced stiffness & balanced frame geometry** — Caltrans SDC 2.1
  **§7.1.2** (Table 7.1.2-1) and **§7.1.3**, AASHTO SGS **§4.1.2** and
  **§4.1.3** — applied at the level each rule actually acts on:
  - *Balanced stiffness* compares bents **inside one frame**: adjacent members
    `≥ 0.75`, any two members `≥ 0.50`, with `κ = kᵉ/m` (or `kᵉ` for the
    constant-width AASHTO form). A run of simply supported spans is a run of
    **single-bent frames**, so no stiffness rule applies there.
  - *Balanced frame geometry* compares **adjacent frames**, `Ti/Tj ≥ 0.70`,
    everywhere. The frame period is the frame's: `T = 2π√(ΣM/ΣK)` over the
    members that resist, on the rigid-deck stand-alone idealisation.
  - Both are evaluated **longitudinally and transversely** and at **every**
    fixity bound like-for-like. The whole feature is behind one tick.
- **Frames are derived per direction** from a `deck_link` column — `integral`
  (monolithic, fixed moment connection), `bearing` (released longitudinally,
  shear key transversely) or `free`. A bearing therefore joins the continuous
  frame transversely and stands alone longitudinally, holding whatever span it
  is fixed to. Tributary weight is entered per direction
  (`weight_long_kip` / `weight_trans_kip`).
- **Column silos** (isolation casings) as the tuning lever — SDC **C7.1.2** /
  SGS **§4.1.4**. A silo of depth `h` lowers the top of shaft and lengthens the
  free column to `H_free = Hcol + h`, softening a stiff pier; the plastic hinge
  stays at the top of shaft (bottom of the silo), the embedded shaft length is
  unchanged, and the p-y springs start at the bottom of the silo. Because
  `H_free` also drives `Lp`, `Δy`, `Δp`, `Vo = Mo/H_free`, the minimum lateral
  strength and P-Δ, **a silo is never free**: the tool auto-sizes silos, re-runs
  the full seismic suite on every changed pier, and iterates until both the
  seismic and the balance checks hold — or reports exactly why it cannot.
- **Batch tabular** workflow (editable table + CSV/Excel import/export), results
  grid, per-column drill-down with M-φ and spectrum plots, and Markdown reports.
- **Point of fixity — assumed *or* soil-derived.** Either the classic Df = 3×/6×
  shaft-diameter multipliers, or an in-house **nonlinear p-y (LPile-equivalent)**
  analysis: the column + Type II shaft are solved as one continuous beam-column
  on layered soil springs (Matlock soft clay, Reese/Welch stiff clay, API/Reese
  sand — cyclic), giving a mechanics-based equivalent depth-to-fixity. Strata are
  entered LPile-style (su, φ′, ε50, k, γ per layer), so a geotech's LPile soil
  table maps in 1:1. Validated against the beam-on-elastic-foundation closed form
  (<0.1 %). A soil too soft / shaft too short for the demand is flagged unstable,
  never credited with a stiff base.
- **Project files** (`.json`) store every column *and* all settings in one file.
  After an optimiser run the optimised column/shaft designs (including bundles)
  are written back into the table, and an in-place **Save** button persists that
  progress to the current project file — no "save as" each time.

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
- **Free column length** `H_free = Hcol + silo` — every mechanics quantity below
  uses `H_free`, not the entered `Hcol`. Without a column silo they are equal.
- **Δy** uses the elastic two-segment cantilever to the point of fixity; **Δp =
  θp·(H_free − Lp/2)** with `θp = Lp·(φu − φy)`; **Δc = Δy + Δp**.
- **Plastic hinge length**: `Lp = 0.08·H_free + 0.15·fye·dbl ≥ 0.3·fye·dbl`.
- **Ultimate confined strain**: `εcu = 0.004 + 1.4·ρs·fyh·εsu / f'cc`.
- **Design spectrum**: AASHTO/Caltrans two-parameter form (`Sds`, `Sd1`);
  displacement demand from the equal-displacement rule at the effective
  (cracked) period.
- **Balance comparison basis**: adjacent piers are compared **bound by bound**
  (stiff-vs-stiff, soft-vs-soft) so the same modelling assumption applies to
  both, and every bound must comply. Note that with `κ = k/m` the identity
  `(Ti/Tj)² = κj/κi` makes the 0.75 stiffness rule stricter than the 0.70 period
  rule (`√0.75 = 0.87`), so §7.1.3 follows from §7.1.2; both are still reported.
- **Column silo and the soil profile**: the silo strips the top `h` of strata
  (the shaft now starts that much deeper) while carrying the removed overburden
  forward, so the near-surface wedge terms restart at the bottom of the silo —
  the conservative, conventional isolation-casing treatment.
- **The balance checks do not change the seismic run.** The per-bent seismic
  suite keeps its fixed-free cantilever and its longitudinal tributary mass.
  Consequently the transverse demand is not checked, and a bent inside a
  continuous frame is still designed as a stand-alone cantilever rather than
  from the frame's displacement demand.
- **Frame stiffness uses the fixed-free member** (`3EcIeff/L³`). That is right
  transversely and for every simply supported bent, but an integral bent is a
  fixed moment connection and so is fixed-fixed *longitudinally*
  (`12EcIeff/L³`, SDC C7.1.2-2). Longitudinal `K_frame` is therefore understated
  and `T_long` overstated for a continuous frame — an approximation the report
  states wherever it prints a frame period.
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
  soil.py              strata + p-y curves (Matlock/Reese/API) + Davisson
  pile_solver.py       FE beam-column on nonlinear Winkler springs (p-y)
  sdc_capacity.py      Lp, Δ-capacity, all SDC checks, shaft capacity protection
  optimizer.py         greedy priority-ordered design search
  balance.py           adjacent-pier balanced stiffness/geometry + silo sizing
  io_schema.py         batch table schema + CSV/Excel I/O + validation
  batch.py             batch runner, balance stages + summary grid
  report.py            per-column and balance Markdown reports
app.py                 Streamlit GUI
tests/                 pytest suite
```
