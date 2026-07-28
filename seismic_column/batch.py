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

from . import G_IN_S2
from .balance import (GEOMETRY_CHECK, STIFFNESS_CHECK, BalanceCriteria,
                      BalanceResult, BentStiffness, adjacent_pairs,
                      balance_checks, bent_stiffness, dedupe, dp_min_silo,
                      joint_feasible, quantise_silo, required_silo,
                      silo_states, stiffness_at_silo)
from .geometry import Geometry
from .io_schema import GlobalConfig, build_soil_profile, in_frame, validate
from .optimizer import ColumnDesign, OptimizeResult, OptimizeSpec, optimize_column
from .provisions import get_provisions
from .sdc_capacity import ColumnAssessment, column_self_weight, evaluate_column


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

    A silo only ever *softens*, so for a failing pair it is always one specific
    pier that must be lengthened — and **both** code criteria can demand it:

    * balanced stiffness — the pier with the larger ``kappa`` comes down to
      ``kappa_other / k_ratio_min``;
    * balanced frame geometry — the pier with the SHORTER period comes down to
      whatever stiffness puts its period at ``T_ratio_min * T_other``, i.e.
      ``k = m * (2*pi / (T_ratio_min * T_other))**2``.

    Both must be driven.  Under mass normalisation the stiffness rule implies
    the period rule (``sqrt(0.75) = 0.87 >= 0.70``) so the second demand never
    binds — but with ``kappa = k`` the two decouple, and the period rule can be
    the only thing failing.  A planner that watched stiffness alone would then
    report "no further silo change is available" while a fix was in reach.
    Note the period rule always carries mass through ``T = 2*pi*sqrt(m/k)``,
    even when mass normalisation is switched off.

    Sizing uses the elastic two-segment cantilever at the pier's current EI
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

    def k_now(b: BentStiffness, bound: int) -> float:
        """Lateral stiffness at the currently planned silo (elastic prediction)."""
        rr = results[b.name]
        geom = Geometry(Hcol=b.Hcol, D_shaft=rr.shaft.D, silo=silos[b.name])
        return geom.lateral_stiffness(rr.assessment.EI_col,
                                      rr.assessment.EI_shaft,
                                      rr.assessment.bounds[bound].multiplier)

    def T_of(b: BentStiffness, k: float) -> float:
        return 2.0 * math.pi * math.sqrt(b.mass / k) if k > 0 else float("nan")

    def soften_to(target_k: float, who: BentStiffness, other: BentStiffness,
                  bound: int, rule: str) -> bool:
        """Plan a silo that brings ``who`` down to ``target_k``.  True if moved."""
        rr = results[who.name]
        geom = Geometry(Hcol=who.Hcol, D_shaft=rr.shaft.D)
        h = required_silo(geom, rr.assessment.EI_col, rr.assessment.EI_shaft,
                          rr.assessment.bounds[bound].multiplier, target_k,
                          silo_min=silos[who.name], silo_max=cap)
        if h is None:                      # the cap binds — go as deep as allowed
            notes.append(
                f"{who.name}: {cfg.max_silo_ft:g} ft silo cap reached and still "
                f"fails {rule} against {other.name} at the {who.label(bound)} "
                f"bound — stiffen {other.name} (larger column), reduce its "
                f"tributary mass, or raise the cap")
            if silos[who.name] < cap:
                silos[who.name] = cap
                return True
            return False
        h = quantise_silo(h, step, cap, floor=silos[who.name])
        if h > silos[who.name] + 1e-9:
            silos[who.name] = h
            return True
        return False

    for _sweep in range(6):
        moved = False
        pairs = adjacent_pairs(bents)
        for bi, bj in list(pairs) + list(reversed(pairs)):
            n_bounds = min(len(bi.k), len(bj.k))
            for bound in range(n_bounds):
                ki, kj = k_now(bi, bound), k_now(bj, bound)
                if not (math.isfinite(ki) and math.isfinite(kj)) or min(ki, kj) <= 0:
                    continue

                # --- balanced stiffness: soften the larger kappa ---
                ai = ki / bi.mass if criteria.mass_normalized else ki
                aj = kj / bj.mass if criteria.mass_normalized else kj
                if min(ai, aj) / max(ai, aj) < criteria.k_ratio_min:
                    stiff, soft = (bi, bj) if ai > aj else (bj, bi)
                    target = min(ai, aj) / criteria.k_ratio_min
                    if criteria.mass_normalized:
                        target *= stiff.mass
                    moved |= soften_to(target, stiff, soft, bound,
                                       "balanced stiffness")
                    ki, kj = k_now(bi, bound), k_now(bj, bound)   # may have moved

                # --- balanced frame geometry: soften the SHORTER period ---
                Ti, Tj = T_of(bi, ki), T_of(bj, kj)
                if not (math.isfinite(Ti) and math.isfinite(Tj)):
                    continue
                if min(Ti, Tj) / max(Ti, Tj) < criteria.T_ratio_min:
                    quick, slow = (bi, bj) if Ti < Tj else (bj, bi)
                    T_target = criteria.T_ratio_min * max(Ti, Tj)
                    target_k = quick.mass * (2.0 * math.pi / T_target) ** 2
                    moved |= soften_to(target_k, quick, slow, bound,
                                       "balanced frame geometry")
        if not moved:
            break
    return silos, dedupe(notes)


def _plan_silos_min(bents: list[BentStiffness], results: dict[str, RowResult],
                    criteria: BalanceCriteria, cfg: GlobalConfig,
                    floors: dict[str, float]) -> tuple[dict[str, float], list[str]]:
    """Minimum-total-silo plan (exact on the buildable grid).

    Builds a predicted stiffness/mass table over every allowed silo depth using
    the elastic two-segment cantilever at the pier's current EI, **calibrated**
    so it reproduces the stiffness actually measured at the pier's current silo.
    That correction matters in soil mode, where lengthening the column also
    re-derives the point of fixity from the p-y solve, so the raw elastic
    formula over-predicts how much a silo softens.  The caller verifies the plan
    with a real analysis and re-calibrates, so a poor prediction costs a pass,
    never a wrong answer.
    """
    cap = cfg.max_silo_ft * 12.0
    step = cfg.silo_step_ft * 12.0
    # States start at the ENTERED floor, not at the current trial depth: a
    # minimiser must be free to take a silo back off once a better-calibrated
    # pass shows less is needed.  (The greedy repair is deliberately monotone;
    # this is not.)
    states = {b.name: silo_states(floors.get(b.name, 0.0), cap, step)
              for b in bents}

    # per-pier calibration factor: measured k / elastic k, at the current silo
    corr: dict[str, float] = {}
    for b in bents:
        rr = results[b.name]
        ke = stiffness_at_silo(Geometry(Hcol=b.Hcol, D_shaft=rr.shaft.D),
                               rr.assessment.EI_col, rr.assessment.EI_shaft,
                               rr.assessment.bounds[0].multiplier, b.silo)
        corr[b.name] = (b.k[0] / ke) if ke > 0 else 1.0

    def k_of(b: BentStiffness, silo: float, bound: int) -> float:
        rr = results[b.name]
        return corr[b.name] * stiffness_at_silo(
            Geometry(Hcol=b.Hcol, D_shaft=rr.shaft.D),
            rr.assessment.EI_col, rr.assessment.EI_shaft,
            rr.assessment.bounds[bound].multiplier, silo)

    def m_of(b: BentStiffness, silo: float) -> float:
        # a deeper silo lengthens the column, so its self-weight participation
        # grows — small, but it lowers kappa and raises T, so ignoring it would
        # make the predictor optimistic
        rr = results[b.name]
        dW = column_self_weight(rr.design.section().Ag, silo - b.silo,
                                cfg.concrete_unit_weight)
        return b.mass + cfg.self_weight_mass_factor * dW / G_IN_S2

    plan = dp_min_silo(bents, states, k_of, m_of, criteria)
    return plan.silos, plan.notes


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

    # With mass normalisation OFF the stiffness rule and the period rule pull on
    # the same k ratio from opposite sides, and a large enough tributary-mass
    # disparity makes them jointly unsatisfiable at ANY stiffness.  Say so up
    # front rather than letting the silo search grind against it.
    for bi, bj in adjacent_pairs(bents):
        ok, mu, (x_lo, x_hi) = joint_feasible(bi, bj, criteria)
        if not ok:
            log.append(
                f"INFEASIBLE — {bi.name}-{bj.name}: tributary masses differ by "
                f"×{max(mu, 1 / mu):.2f} ({bi.mass:.2f} vs {bj.mass:.2f} "
                f"kip·s²/in). With mass normalisation off, balanced stiffness "
                f"needs k ratio ≥ {criteria.k_ratio_min:.2f} while balanced "
                f"geometry needs it ≤ {mu / criteria.T_ratio_min ** 2:.2f} — no "
                f"stiffness satisfies both, so no silo can fix this pair. "
                f"Rebalance the tributary spans, or switch mass normalisation "
                f"on (the Caltrans form), which makes the period rule "
                f"automatic.")
        else:
            # how much of the unconstrained stiffness window the mass disparity
            # has eaten; below ~60% the pair is the sensitive one on the bridge
            full = 1.0 / criteria.k_ratio_min - criteria.k_ratio_min
            if full > 0 and (x_hi - x_lo) / full < 0.6:
                log.append(
                    f"TIGHT — {bi.name}-{bj.name}: tributary masses differ by "
                    f"×{max(mu, 1 / mu):.2f}, leaving only k ratio "
                    f"[{x_lo:.3f}, {x_hi:.3f}] to satisfy both clauses at once "
                    f"(vs [{criteria.k_ratio_min:.3f}, "
                    f"{1 / criteria.k_ratio_min:.3f}] on a mass-matched pair) — "
                    f"expect this pair to be the hard one to tune.")
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

    minimise = cfg.balance_strategy == "min_silo"
    planner = _plan_silos_min if minimise else _plan_silos
    if minimise:
        log.append("Silo strategy: minimum total depth (exact on the buildable "
                   f"{cfg.silo_step_ft:g} ft grid, per frame), verified against "
                   "a real analysis each pass.")
    # the cheapest FEASIBLE state seen, so a later refinement pass can never
    # leave the run worse off than one it already had
    best: tuple[float, list[RowResult], list, list] | None = None

    for outer in range(1, max(cfg.balance_max_outer, 1) + 1):
        silos, notes = planner(bents, by_name, criteria, cfg, floors)
        changed = [n for n, h in silos.items()
                   if abs(h - by_name[n].silo) > 1e-6]
        log.extend(n for n in notes if n not in log)   # same cap note each pass
        if not changed:
            if result.passed:
                log.append(f"Pass {outer}: the plan is unchanged — settled at "
                           f"{sum(r.silo for r in results) / 12:g} ft total.")
                result.converged = True
            else:
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
            total = sum(r.silo for r in results)
            if best is None or total < best[0] - 1e-9:
                best = (total, list(results), list(bents), list(checks))
            log.append(f"Pass {outer}: balanced at {total / 12:g} ft of silo "
                       f"over {sum(1 for r in results if r.silo > 0)} pier(s).")
            if not minimise:
                result.converged = True
                break
            # keep going: the next pass re-calibrates the predictor against the
            # stiffness just measured, which can find a cheaper plan.  The loop
            # exits above when the plan stops changing.
            result.converged = True
    else:
        if not result.passed:
            log.append(
                f"Still unbalanced after {cfg.balance_max_outer} passes — the "
                "silo and the seismic re-design are fighting each other (a "
                "deeper silo raises the demands, which grows the section, which "
                "stiffens it again). Stiffen the flexible pier or relax the "
                "geometry.")
            result.converged = False

    if best is not None:
        # restore the cheapest feasible state, in case a refinement pass ended
        # somewhere more expensive or broke feasibility outright
        total_now = sum(r.silo for r in results)
        if not result.passed or total_now > best[0] + 1e-9:
            results[:] = best[1]
            result.bents, result.checks = best[2], best[3]
            by_name = {rr.name: rr for rr in results}
            log.append(f"Kept the cheapest feasible plan found: "
                       f"{best[0] / 12:g} ft of silo.")
        result.converged = True

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
        # the length the checks were actually run at (Hcol + silo)
        "H_free_ft": round(a.H_free / 12.0, 2),
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
