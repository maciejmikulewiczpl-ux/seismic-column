"""Batch runner: analyse/optimise every column in a tabular input.

Runs in up to three stages:

1. every row is optimised/evaluated independently against the seismic checks
   (this is all the tool did before balance checking existed);
2. adjacent piers are checked for balanced stiffness and balanced frame geometry
   (see :mod:`seismic_column.balance`);
3. when they fail, column silos are sized to soften the stiff piers and the
   affected rows are **re-run through the full seismic suite** — a longer free
   column changes Lp, the displacement demand, Vo, Vp and P-Delta, so a silo is
   never free.  That re-run can change the reinforcement, which changes EI and
   hence the stiffness, so stages 2-3 iterate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .balance import (GEOMETRY_CHECK, STIFFNESS_CHECK, BalanceCriteria,
                      BalanceResult, BentStiffness, adjacent_pairs,
                      balance_checks, bent_stiffness, dedupe, quantise_silo,
                      required_silo)
from .geometry import Geometry
from .io_schema import GlobalConfig, build_soil_profile, in_frame, validate
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
    frame: str = ""
    silo: float = 0.0          # column silo depth actually analysed, in


@dataclass
class BatchOutcome:
    """Everything a batch run produced."""

    summary: pd.DataFrame
    results: list[RowResult]
    balance: BalanceResult | None = None   # None when the balance checks are off


def _bundle(row: pd.Series, key: str) -> int:
    """Read a bundle-count column, defaulting to 1 for older tables/NaN."""
    val = row.get(key, 1)
    if val is None or pd.isna(val):
        return 1
    return max(1, int(val))


def _row_to_inputs(row: pd.Series, cfg: GlobalConfig, silo: float | None = None):
    """Build the analysis inputs for one row.

    ``silo`` (in) overrides the row's ``silo_ft`` so the balance stage can re-run
    a pier at a trial silo depth without rewriting the table.
    """
    _prov = get_provisions(cfg.code)
    fce_factor, fce_floor = _prov.fce_factor, _prov.fce_floor
    row_silo = float(row.get("silo_ft", 0.0) or 0.0) * 12.0
    geometry = Geometry(Hcol=float(row["Hcol_ft"]) * 12.0,
                        D_shaft=float(row["D_shaft_in"]),
                        silo=row_silo if silo is None else float(silo))
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


def run_row(row: pd.Series, cfg: GlobalConfig, on_candidate=None,
            silo: float | None = None) -> RowResult:
    """Run a single batch row (optimise or evaluate).

    ``on_candidate(iters)`` (optional) is forwarded to the optimiser for live
    within-column progress (a soil p-y optimise can take a while per column).
    ``silo`` (in) overrides the row's entered silo depth — used by the balance
    stage to re-analyse a pier at a trial depth.
    """
    geometry, column, shaft, mults = _row_to_inputs(row, cfg, silo=silo)
    frame = str(row.get("frame", "") or "")
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
                         True, res.log, frame=frame, silo=geometry.silo)

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
    return RowResult(name, column, shaft, assessment, assessment.passed, False, [],
                     frame=frame, silo=geometry.silo)


def _criteria(cfg: GlobalConfig) -> BalanceCriteria:
    """Balance limits for this run — a user entry may only be *stricter*."""
    prov = get_provisions(cfg.code)
    return BalanceCriteria(
        k_ratio_min=max(cfg.balance_k_ratio_min, prov.balance_k_ratio_adjacent),
        T_ratio_min=max(cfg.balance_T_ratio_min, prov.balance_T_ratio),
        mass_normalized=cfg.balance_mass_normalized,
        ref_stiffness=prov.ref_balanced_stiffness,
        ref_geometry=prov.ref_balanced_geometry,
    )


def _build_bents(results: list[RowResult], order: dict[str, int]
                 ) -> list[BentStiffness]:
    """Bents that take part in the checks, in table row order."""
    return [
        bent_stiffness(rr.name, rr.frame, order[rr.name], rr.assessment,
                       Hcol=rr.assessment.Hcol_entered, silo=rr.silo)
        for rr in results
        if in_frame(rr.frame) and rr.assessment.bounds
    ]


def run_batch(df: pd.DataFrame, cfg: GlobalConfig,
              progress=None, on_candidate=None) -> tuple[pd.DataFrame, list[RowResult]]:
    """Run the whole batch; return (summary DataFrame, list of RowResult).

    Thin wrapper over :func:`run_batch_balanced` for callers that only want the
    per-column results.  See that function for the balance stages.
    """
    outcome = run_batch_balanced(df, cfg, progress=progress,
                                 on_candidate=on_candidate)
    return outcome.summary, outcome.results


def run_batch_balanced(df: pd.DataFrame, cfg: GlobalConfig,
                       progress=None, on_candidate=None,
                       balance_progress=None) -> BatchOutcome:
    """Run the whole batch, including the adjacent-pier balance stages.

    ``progress`` (optional) is a callback invoked after each column with
    ``(done, total, name, status)`` — used by the GUI to show a live progress
    bar so long soil/optimiser runs don't look like a crash.  ``on_candidate``
    (optional) ``(name, iters)`` fires per trial design *within* a column, so a
    single slow (soil p-y) column also shows live movement.

    ``progress`` covers stage 1 only — it fires exactly once per table row, with
    a done-count that grows to ``(total, total)``.  The balance stage re-runs an
    unpredictable subset of rows an unpredictable number of times, so it reports
    through the separate ``balance_progress(message)`` callback instead of
    corrupting that count.

    Stage 1 runs every row independently.  When ``cfg.balance_check`` is set,
    stage 2 checks adjacent piers for balanced stiffness / balanced frame
    geometry and stage 3 (``cfg.balance_auto_silo``) sizes column silos and
    re-runs the affected rows until both the seismic and the balance checks hold
    or the iteration cap / silo cap binds.
    """
    # NB no ``max_silo_ft`` here: that cap governs how deep the tool may size a
    # silo by itself.  A deeper silo typed into the table is the engineer's call
    # and is honoured as-is (and as a floor the auto-siloer never reduces).
    df = validate(df, get_provisions(cfg.code).min_shaft_oversize,
                  optimize=cfg.optimize)
    total = len(df)
    results: list[RowResult] = []
    rows_by_name: dict[str, pd.Series] = {}
    order: dict[str, int] = {}
    # (name | error-dict) in table order, so the summary keeps row order even
    # when a row blows up mid-batch.  Names, not results: the balance stage
    # replaces RowResult objects, and the summary must show what it finished on.
    slots: list[object] = []
    for i, (_, row) in enumerate(df.iterrows()):
        name = str(row.get("name", "?"))
        row_cb = (lambda it, _n=name: on_candidate(_n, it)) if on_candidate else None
        try:
            rr = run_row(row, cfg, on_candidate=row_cb)
        except Exception as exc:  # keep the batch going, flag the row
            slots.append({"name": name, "status": f"ERROR: {exc}",
                          "feasible": False})
            if progress is not None:
                progress(i + 1, total, name, "ERROR")
            continue
        results.append(rr)
        slots.append(name)
        rows_by_name[name] = row
        order[name] = i
        if progress is not None:
            progress(i + 1, total, name, "PASS" if rr.feasible else "FAIL")

    balance = None
    if cfg.balance_check:
        balance = _balance_stage(results, rows_by_name, order, cfg,
                                 on_candidate=on_candidate,
                                 on_status=balance_progress)

    final = {rr.name: rr for rr in results}      # post-balance designs
    summary = pd.DataFrame([
        s if isinstance(s, dict) else _summary_row(final[s], balance)
        for s in slots])
    return BatchOutcome(summary=summary, results=results, balance=balance)


def _plan_silos(bents: list[BentStiffness], results: dict[str, RowResult],
                criteria: BalanceCriteria, cfg: GlobalConfig,
                floors: dict[str, float]) -> tuple[dict[str, float], list[str]]:
    """Silo depths (in) that would bring every adjacent pair into compliance.

    A silo only ever *softens*, so for a failing pair it is the stiffer pier that
    gets lengthened, down to ``kappa_soft / k_ratio_min``.  Sizing uses the
    elastic two-segment cantilever at the pier's current EI
    (:func:`~seismic_column.balance.required_silo`), which is free — the caller
    then pays for one real re-analysis per changed pier and iterates, because
    the re-analysis may change the reinforcement and hence EI.

    Sweeps forward then backward until the depths stop moving: softening a pier
    toward its neighbour is monotone, and rounding up to ``silo_step_ft`` bounds
    any overshoot past a neighbour on the other side.

    This is only a *predictor*.  It holds EI and the fixity multiplier fixed —
    in soil mode the multiplier is itself derived from the p-y solve at the
    current silo, so the prediction is roughest there.  The caller always
    verifies with a real re-analysis and loops, so a poor prediction costs an
    extra pass, never a wrong answer.
    """
    cap = cfg.max_silo_ft * 12.0
    step = cfg.silo_step_ft * 12.0
    silos = {b.name: max(b.silo, floors.get(b.name, 0.0)) for b in bents}
    notes: list[str] = []

    def kappa_now(b: BentStiffness, bound: int) -> float:
        """kappa at the currently planned silo depth (elastic prediction)."""
        rr = results[b.name]
        geom = Geometry(Hcol=b.Hcol, D_shaft=rr.shaft.D, silo=silos[b.name])
        k = geom.lateral_stiffness(rr.assessment.EI_col, rr.assessment.EI_shaft,
                                   rr.assessment.bounds[bound].multiplier)
        return k / b.mass if criteria.mass_normalized else k

    for _sweep in range(6):
        moved = False
        pairs = adjacent_pairs(bents)
        for bi, bj in list(pairs) + list(reversed(pairs)):
            n_bounds = min(len(bi.k), len(bj.k))
            for bound in range(n_bounds):
                ki, kj = kappa_now(bi, bound), kappa_now(bj, bound)
                if not (math.isfinite(ki) and math.isfinite(kj)) or min(ki, kj) <= 0:
                    continue
                if min(ki, kj) / max(ki, kj) >= criteria.k_ratio_min:
                    continue
                stiff, soft = (bi, bj) if ki > kj else (bj, bi)
                k_soft = min(ki, kj)
                # target on the STIFF pier, converted back to a stiffness
                target = k_soft / criteria.k_ratio_min
                rr = results[stiff.name]
                if criteria.mass_normalized:
                    target *= stiff.mass
                geom = Geometry(Hcol=stiff.Hcol, D_shaft=rr.shaft.D)
                h = required_silo(
                    geom, rr.assessment.EI_col, rr.assessment.EI_shaft,
                    rr.assessment.bounds[bound].multiplier, target,
                    silo_min=silos[stiff.name], silo_max=cap)
                if h is None:
                    # the cap binds — go as deep as allowed and report it
                    if silos[stiff.name] < cap:
                        silos[stiff.name] = cap
                        moved = True
                    notes.append(
                        f"{stiff.name}: {cfg.max_silo_ft:g} ft silo cap reached "
                        f"and still stiffer than {soft.name} at the "
                        f"{stiff.label(bound)} bound — stiffen {soft.name} "
                        f"(larger column) or raise the cap")
                    continue
                h = quantise_silo(h, step, cap, floor=silos[stiff.name])
                if h > silos[stiff.name] + 1e-9:
                    silos[stiff.name] = h
                    moved = True
        if not moved:
            break
    return silos, dedupe(notes)


def _balance_stage(results: list[RowResult], rows_by_name: dict[str, pd.Series],
                   order: dict[str, int], cfg: GlobalConfig,
                   on_candidate=None, on_status=None) -> BalanceResult:
    """Check adjacent-pier balance and, if enabled, auto-size column silos.

    Returns the final :class:`~seismic_column.balance.BalanceResult`; ``results``
    is mutated in place so its entries always describe the design that was
    finally checked.
    """
    criteria = _criteria(cfg)
    bents = _build_bents(results, order)
    checks = balance_checks(bents, criteria)
    log: list[str] = []
    if len(bents) < 2:
        log.append("Fewer than two piers take part in the balance checks "
                   "(check the 'frame' column) — nothing to compare.")
    result = BalanceResult(bents=bents, checks=checks, criteria=criteria,
                           log=log, converged=True)
    if result.passed or not cfg.balance_auto_silo or len(bents) < 2:
        if result.passed and len(bents) >= 2:
            log.append("All adjacent pairs comply as designed — no silo needed.")
        elif not cfg.balance_auto_silo and not result.passed:
            log.append("Auto-silo is off: reporting the shortfall without "
                       "changing any design.")
        return result

    by_name = {rr.name: rr for rr in results}
    floors = {rr.name: float(rows_by_name[rr.name].get("silo_ft", 0.0) or 0.0) * 12.0
              for rr in results}

    for outer in range(1, max(cfg.balance_max_outer, 1) + 1):
        silos, notes = _plan_silos(bents, by_name, criteria, cfg, floors)
        changed = [n for n, h in silos.items()
                   if abs(h - by_name[n].silo) > 1e-6]
        log.extend(n for n in notes if n not in log)   # same cap note each pass
        if not changed:
            log.append(f"Pass {outer}: no further silo change is available — "
                       "stopping.")
            result.converged = False
            break
        for name in changed:
            old = by_name[name].silo
            row_cb = ((lambda it, _n=name: on_candidate(_n, it))
                      if on_candidate else None)
            if on_status is not None:
                on_status(f"Balancing pass {outer}: re-analysing {name} with a "
                          f"{silos[name] / 12.0:.1f} ft silo…")
            try:
                rr = run_row(rows_by_name[name], cfg, on_candidate=row_cb,
                             silo=silos[name])
            except Exception as exc:
                log.append(f"Pass {outer}: {name} failed to re-analyse at a "
                           f"{silos[name] / 12.0:.1f} ft silo ({exc}) — silo "
                           "left unchanged.")
                continue
            log.append(
                f"Pass {outer}: {name} silo {old / 12.0:.1f} → "
                f"{silos[name] / 12.0:.1f} ft (free length "
                f"{rr.assessment.H_free / 12.0:.1f} ft); seismic re-check "
                f"{'PASS' if rr.feasible else 'FAIL'}")
            by_name[name] = rr
            for idx, existing in enumerate(results):
                if existing.name == name:
                    results[idx] = rr
                    break

        bents = _build_bents(results, order)
        checks = balance_checks(bents, criteria)
        result.bents, result.checks = bents, checks
        if result.passed:
            log.append(f"Balanced after {outer} silo pass"
                       f"{'es' if outer > 1 else ''}.")
            result.converged = True
            break
    else:
        log.append(f"Still unbalanced after {cfg.balance_max_outer} passes — "
                   "the silo and the seismic re-design are fighting each other "
                   "(a deeper silo raises the demands, which grows the section, "
                   "which stiffens it again). Stiffen the flexible pier or "
                   "relax the geometry.")
        result.converged = False

    if not result.passed:
        for c in result.failed:
            log.append(f"UNRESOLVED — {c.label}: ratio {c.ratio:.3f} "
                       f"< {c.limit:.2f} ({c.note})")
    return result


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

    Rows are matched by ``name``.  Geometry, reinforcement, bundle columns and
    the column silo depth are overwritten with the design carried by each result
    so the table becomes the current design of record; loads, entered height,
    frame, spectrum-independent inputs and fixity multipliers are left
    untouched.  Rows with no matching result (e.g. a run error) are left as-is.
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
        # the balance stage may have deepened the silo — that is a design output
        df.at[i, "silo_ft"] = rr.silo / 12.0
    return validate(df, optimize=True)


def _summary_row(rr: RowResult, balance: BalanceResult | None = None) -> dict:
    a = rr.assessment
    g = a.governing_bound
    d = rr.design
    row = {
        "name": rr.name,
        "feasible": rr.feasible,
        "status": "PASS" if rr.feasible else "FAIL",
        "silo_ft": round(rr.silo / 12.0, 2),
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
        "k_e_kip_in": round(g.stiffness, 2),
        "T_eff_s": round(g.demand.period, 3),
        "checks_failed": "; ".join(c.name for c in a.checks if not c.passed) or "-",
    }
    if balance is not None:
        rk = balance.worst_ratio(rr.name, STIFFNESS_CHECK)
        rt = balance.worst_ratio(rr.name, GEOMETRY_CHECK)
        row.update({
            "frame": rr.frame,
            "bal_k_ratio": round(rk, 3) if rk is not None else None,
            "bal_T_ratio": round(rt, 3) if rt is not None else None,
            "balanced": ("-" if rk is None and rt is None
                         else ("PASS" if balance.pier_passed(rr.name) else "FAIL")),
        })
    return row
