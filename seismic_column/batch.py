"""Batch runner: analyse/optimise every column in a tabular input."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .geometry import Geometry
from .io_schema import GlobalConfig, build_soil_profile, validate
from .optimizer import ColumnDesign, OptimizeResult, OptimizeSpec, optimize_column
from .provisions import get_provisions
from .sdc_capacity import ColumnAssessment, evaluate_column


@dataclass
class RowResult:
    name: str
    design: ColumnDesign
    shaft: ColumnDesign
    assessment: ColumnAssessment
    feasible: bool
    optimized: bool
    log: list[str]


def _bundle(row: pd.Series, key: str) -> int:
    """Read a bundle-count column, defaulting to 1 for older tables/NaN."""
    val = row.get(key, 1)
    if val is None or pd.isna(val):
        return 1
    return max(1, int(val))


def _row_to_inputs(row: pd.Series, cfg: GlobalConfig):
    _prov = get_provisions(cfg.code)
    fce_factor, fce_floor = _prov.fce_factor, _prov.fce_floor
    geometry = Geometry(Hcol=float(row["Hcol_ft"]) * 12.0,
                        D_shaft=float(row["D_shaft_in"]))
    column = ColumnDesign(
        D=float(row["Dcol_in"]), fc=float(row["fc_ksi"]), cover=float(row["cover_in"]),
        n_bars=int(row["n_bars"]), long_bar_no=int(row["long_bar_no"]),
        long_bundle=_bundle(row, "long_bundle"),
        spiral_bar_no=int(row["spiral_bar_no"]),
        spiral_spacing=float(row["spiral_spacing_in"]),
        spiral_bundle=_bundle(row, "spiral_bundle"),
        fye=cfg.fye, fue=cfg.fue, fyh=cfg.fyh, fce_factor=fce_factor, fce_floor=fce_floor,
    )
    shaft = ColumnDesign(
        D=float(row["D_shaft_in"]), fc=float(row["shaft_fc_ksi"]),
        cover=float(row["shaft_cover_in"]), n_bars=int(row["shaft_n_bars"]),
        long_bar_no=int(row["shaft_long_bar_no"]),
        long_bundle=_bundle(row, "shaft_long_bundle"),
        spiral_bar_no=int(row["shaft_spiral_bar_no"]),
        spiral_spacing=float(row["shaft_spiral_spacing_in"]),
        spiral_bundle=_bundle(row, "shaft_spiral_bundle"),
        fye=cfg.fye, fue=cfg.fue, fyh=cfg.fyh, fce_factor=fce_factor, fce_floor=fce_floor,
    )
    mults = (float(row["mult_lb"]), float(row["mult_ub"]))
    return geometry, column, shaft, mults


def run_row(row: pd.Series, cfg: GlobalConfig, on_candidate=None) -> RowResult:
    """Run a single batch row (optimise or evaluate).

    ``on_candidate(iters)`` (optional) is forwarded to the optimiser for live
    within-column progress (a soil p-y optimise can take a while per column).
    """
    geometry, column, shaft, mults = _row_to_inputs(row, cfg)
    spectrum = cfg.design_spectrum.build()
    lle_spectrum = cfg.lle_spectrum.build() if cfg.lle_spectrum else None
    provisions = get_provisions(cfg.code)
    # The selected code's longitudinal limits govern; a user entry may only be
    # *stricter*.  Without this the provisions values were never read at all and
    # the GUI could silently accept rho_l below the code minimum.
    rho_l_min = max(cfg.rho_l_min, provisions.rho_l_min)
    rho_l_max = min(cfg.rho_l_max, provisions.rho_l_max)
    mu_d_limit = min(cfg.mu_d_limit, provisions.mu_d_limit_single)
    axial = float(row["axial_kip"])
    weight = float(row["weight_kip"])
    name = str(row["name"])

    # soil-structure interaction (point of fixity from p-y strata)
    soil_profile = build_soil_profile(cfg) if cfg.fixity_source == "soil" else None
    soil_kw = dict(
        fixity_source=cfg.fixity_source, soil_profile=soil_profile,
        shaft_embed_length=cfg.shaft_embed_ft * 12.0,
        soil_bounds=(cfg.soil_stiff_factor, cfg.soil_soft_factor),
    )

    if cfg.optimize:
        spec = OptimizeSpec(
            variable=set(cfg.variable), priority=tuple(cfg.priority),
            rho_l_min=rho_l_min, rho_l_max=rho_l_max,
            min_bar_spacing=cfg.min_bar_spacing, allow_bundling=cfg.allow_bundling,
            min_shaft_oversize=cfg.min_shaft_oversize_in,
            objective=cfg.optimize_objective, target_rho=cfg.target_rho_l,
        )
        res: OptimizeResult = optimize_column(
            column, shaft, geometry, spectrum, axial, weight, spec=spec,
            fixity_multipliers=mults, shaft_moment_basis=cfg.shaft_moment_basis,
            lle_spectrum=lle_spectrum, lle_mu_limit=cfg.lle_mu_limit,
            concrete_unit_weight=cfg.concrete_unit_weight,
            self_weight_mass_factor=cfg.self_weight_mass_factor,
            self_weight_in_axial=cfg.self_weight_in_axial,
            provisions=provisions, on_candidate=on_candidate, **soil_kw,
        )
        return RowResult(name, res.design, res.shaft, res.assessment, res.feasible,
                        True, res.log)

    assessment = evaluate_column(
        column.section(), shaft.section(), geometry, spectrum, axial, weight,
        fixity_multipliers=mults, mu_d_limit=mu_d_limit,
        rho_l_min=rho_l_min, rho_l_max=rho_l_max,
        shaft_moment_basis=cfg.shaft_moment_basis,
        lle_spectrum=lle_spectrum, lle_mu_limit=cfg.lle_mu_limit,
        concrete_unit_weight=cfg.concrete_unit_weight,
        self_weight_mass_factor=cfg.self_weight_mass_factor,
        self_weight_in_axial=cfg.self_weight_in_axial,
        provisions=provisions, **soil_kw,
    )
    return RowResult(name, column, shaft, assessment, assessment.passed, False, [])


def run_batch(df: pd.DataFrame, cfg: GlobalConfig,
              progress=None, on_candidate=None) -> tuple[pd.DataFrame, list[RowResult]]:
    """Run the whole batch; return (summary DataFrame, list of RowResult).

    ``progress`` (optional) is a callback invoked after each column with
    ``(done, total, name, status)`` — used by the GUI to show a live progress
    bar so long soil/optimiser runs don't look like a crash.  ``on_candidate``
    (optional) ``(name, iters)`` fires per trial design *within* a column, so a
    single slow (soil p-y) column also shows live movement.
    """
    df = validate(df, get_provisions(cfg.code).min_shaft_oversize,
                  optimize=cfg.optimize)
    total = len(df)
    results: list[RowResult] = []
    summary_rows: list[dict] = []
    for i, (_, row) in enumerate(df.iterrows()):
        name = str(row.get("name", "?"))
        row_cb = (lambda it, _n=name: on_candidate(_n, it)) if on_candidate else None
        try:
            rr = run_row(row, cfg, on_candidate=row_cb)
        except Exception as exc:  # keep the batch going, flag the row
            summary_rows.append({
                "name": name, "status": f"ERROR: {exc}", "feasible": False,
            })
            if progress is not None:
                progress(i + 1, total, name, "ERROR")
            continue
        results.append(rr)
        summary_rows.append(_summary_row(rr))
        if progress is not None:
            progress(i + 1, total, name, "PASS" if rr.feasible else "FAIL")
    return pd.DataFrame(summary_rows), results


# Batch-table column <- ColumnDesign attribute, for the column and the shaft.
_COL_DESIGN_MAP = {
    "Dcol_in": "D", "fc_ksi": "fc", "cover_in": "cover", "n_bars": "n_bars",
    "long_bar_no": "long_bar_no", "long_bundle": "long_bundle",
    "spiral_bar_no": "spiral_bar_no", "spiral_spacing_in": "spiral_spacing",
    "spiral_bundle": "spiral_bundle",
}
_SHAFT_DESIGN_MAP = {
    "D_shaft_in": "D",
    "shaft_fc_ksi": "fc", "shaft_cover_in": "cover", "shaft_n_bars": "n_bars",
    "shaft_long_bar_no": "long_bar_no", "shaft_long_bundle": "long_bundle",
    "shaft_spiral_bar_no": "spiral_bar_no",
    "shaft_spiral_spacing_in": "spiral_spacing",
    "shaft_spiral_bundle": "spiral_bundle",
}


def results_to_dataframe(results: list[RowResult],
                         base_df: pd.DataFrame) -> pd.DataFrame:
    """Write the (optimised) column + shaft designs back into the batch table.

    Rows are matched by ``name``.  Geometry, reinforcement and bundle columns
    are overwritten with the design carried by each result so the table becomes
    the current design of record; loads, height, spectrum-independent inputs
    and fixity multipliers are left untouched.  Rows with no matching result
    (e.g. a run error) are left as-is.
    """
    # write-back only happens for an optimise run, whose input may have left the
    # rebar blank — tolerate that (optimize=True), the results overwrite it.
    df = validate(base_df, optimize=True).copy()
    by_name = {r.name: r for r in results}
    for i, name in df["name"].items():
        rr = by_name.get(str(name))
        if rr is None:
            continue
        for col, attr in _COL_DESIGN_MAP.items():
            df.at[i, col] = getattr(rr.design, attr)
        for col, attr in _SHAFT_DESIGN_MAP.items():
            df.at[i, col] = getattr(rr.shaft, attr)
    return validate(df, optimize=True)


def _summary_row(rr: RowResult) -> dict:
    a = rr.assessment
    g = a.governing_bound
    d = rr.design
    return {
        "name": rr.name,
        "feasible": rr.feasible,
        "status": "PASS" if rr.feasible else "FAIL",
        "Dcol_in": d.D,
        "fc_ksi": d.fc,
        "long": d.long_label(),
        "rho_l_%": round(d.rho_l() * 100.0, 2),
        "spiral": d.spiral_label(),
        "rho_s_%": round(d.section().rho_s * 100.0, 2),
        "shaft_long": rr.shaft.long_label(),
        "shaft_spiral": rr.shaft.spiral_label(),
        "Mp_kft": round(a.mc_col.Mp / 12.0, 0),
        "phi_u": a.mc_col.phi_u,
        "Ieff/Ig_col": round(a.Ieff_col / a.Ig_col, 3),
        "Ieff/Ig_shaft": round(a.Ieff_shaft / a.Ig_shaft, 3),
        "Dd_in": round(g.demand.disp_demand, 2),
        "Dc_in": round(g.delta_c, 2),
        "Dc/Dd": round(g.delta_c / g.demand.disp_demand, 2) if g.demand.disp_demand else None,
        "mu_d": round(max(b.mu_demand for b in a.bounds), 2),
        "mu_LLE": (round(max(b.mu_lle for b in a.bounds), 2)
                   if a.bounds[0].mu_lle is not None else None),
        "checks_failed": "; ".join(c.name for c in a.checks if not c.passed) or "-",
    }
