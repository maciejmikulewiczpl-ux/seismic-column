"""Batch runner: analyse/optimise every column in a tabular input."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .geometry import Geometry
from .io_schema import GlobalConfig, validate
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


def _row_to_inputs(row: pd.Series, cfg: GlobalConfig):
    geometry = Geometry(Hcol=float(row["Hcol_ft"]) * 12.0,
                        D_shaft=float(row["D_shaft_in"]))
    column = ColumnDesign(
        D=float(row["Dcol_in"]), fc=float(row["fc_ksi"]), cover=float(row["cover_in"]),
        n_bars=int(row["n_bars"]), long_bar_no=int(row["long_bar_no"]),
        spiral_bar_no=int(row["spiral_bar_no"]),
        spiral_spacing=float(row["spiral_spacing_in"]),
        fye=cfg.fye, fue=cfg.fue, fyh=cfg.fyh,
    )
    shaft = ColumnDesign(
        D=float(row["D_shaft_in"]), fc=float(row["shaft_fc_ksi"]),
        cover=float(row["shaft_cover_in"]), n_bars=int(row["shaft_n_bars"]),
        long_bar_no=int(row["shaft_long_bar_no"]),
        spiral_bar_no=int(row["shaft_spiral_bar_no"]),
        spiral_spacing=float(row["shaft_spiral_spacing_in"]),
        fye=cfg.fye, fue=cfg.fue, fyh=cfg.fyh,
    )
    mults = (float(row["mult_lb"]), float(row["mult_ub"]))
    return geometry, column, shaft, mults


def run_row(row: pd.Series, cfg: GlobalConfig) -> RowResult:
    """Run a single batch row (optimise or evaluate)."""
    geometry, column, shaft, mults = _row_to_inputs(row, cfg)
    spectrum = cfg.design_spectrum.build()
    lle_spectrum = cfg.lle_spectrum.build() if cfg.lle_spectrum else None
    provisions = get_provisions(cfg.code)
    axial = float(row["axial_kip"])
    weight = float(row["weight_kip"])
    name = str(row["name"])

    if cfg.optimize:
        spec = OptimizeSpec(
            variable=set(cfg.variable), priority=tuple(cfg.priority),
            rho_l_min=cfg.rho_l_min, rho_l_max=cfg.rho_l_max,
            min_bar_spacing=cfg.min_bar_spacing, allow_bundling=cfg.allow_bundling,
        )
        res: OptimizeResult = optimize_column(
            column, shaft, geometry, spectrum, axial, weight, spec=spec,
            fixity_multipliers=mults, shaft_moment_basis=cfg.shaft_moment_basis,
            lle_spectrum=lle_spectrum, lle_mu_limit=cfg.lle_mu_limit,
            concrete_unit_weight=cfg.concrete_unit_weight,
            self_weight_mass_factor=cfg.self_weight_mass_factor,
            self_weight_in_axial=cfg.self_weight_in_axial,
            provisions=provisions,
        )
        return RowResult(name, res.design, res.shaft, res.assessment, res.feasible,
                        True, res.log)

    assessment = evaluate_column(
        column.section(), shaft.section(), geometry, spectrum, axial, weight,
        fixity_multipliers=mults, mu_d_limit=cfg.mu_d_limit,
        rho_l_min=cfg.rho_l_min, rho_l_max=cfg.rho_l_max,
        shaft_moment_basis=cfg.shaft_moment_basis,
        lle_spectrum=lle_spectrum, lle_mu_limit=cfg.lle_mu_limit,
        concrete_unit_weight=cfg.concrete_unit_weight,
        self_weight_mass_factor=cfg.self_weight_mass_factor,
        self_weight_in_axial=cfg.self_weight_in_axial,
        provisions=provisions,
    )
    return RowResult(name, column, shaft, assessment, assessment.passed, False, [])


def run_batch(df: pd.DataFrame, cfg: GlobalConfig) -> tuple[pd.DataFrame, list[RowResult]]:
    """Run the whole batch; return (summary DataFrame, list of RowResult)."""
    df = validate(df)
    results: list[RowResult] = []
    summary_rows: list[dict] = []
    for _, row in df.iterrows():
        try:
            rr = run_row(row, cfg)
        except Exception as exc:  # keep the batch going, flag the row
            summary_rows.append({
                "name": str(row.get("name", "?")), "status": f"ERROR: {exc}",
                "feasible": False,
            })
            continue
        results.append(rr)
        summary_rows.append(_summary_row(rr))
    return pd.DataFrame(summary_rows), results


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
