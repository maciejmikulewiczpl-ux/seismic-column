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

import itertools
import math
from dataclasses import dataclass, field, replace

import pandas as pd

from . import G_IN_S2
from .balance import (GEOMETRY_CHECK, STIFFNESS_ANY_CHECK, STIFFNESS_CHECK,
                      BalanceCriteria, BalanceResult, BentStiffness,
                      adjacent_pairs, balance_checks, bent_stiffness, dedupe,
                      dp_min_silo, frames_for, quantise_silo,
                      required_silo, silo_states, stiffness_at_silo)
from .bent import BentAssessment, evaluate_bent
from .frame_seismic import FrameCheck, check_all
from .geometry import Geometry
from .io_schema import (DIRECTIONS, LONGITUDINAL, TRANSVERSE, GlobalConfig,
                         build_soil_profile, in_frame, validate)
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
    deck_link: str = "integral"   # integral | bearing | free
    weight_trans: float = 0.0     # entered TRANSVERSE tributary weight, kip
    # A bent may carry more than one column.  ``assessment`` above stays the
    # GOVERNING column so every existing consumer keeps working; ``bent`` holds
    # the positions and the transverse push/pull between them.
    bent: BentAssessment | None = None
    cap_fixity: str = "fixed"     # transverse column-to-cap detail

    @property
    def n_columns(self) -> int:
        return self.bent.n_columns if self.bent else 1

    @property
    def columns(self) -> list:
        return list(self.bent.positions) if self.bent else []


@dataclass
class BatchOutcome:
    """Everything a batch run produced."""

    summary: pd.DataFrame
    results: list[RowResult]
    balance: BalanceResult | None = None   # None when the balance checks are off
    # Frame-level displacement check for the continuous frames, at each member's
    # REAL end condition.  Empty unless the balance checks ran (it needs the
    # derived frames) and a continuous frame exists.
    frame_checks: list[FrameCheck] = field(default_factory=list)


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
            silo: float | None = None, demand_basis: dict | None = None,
            end_fixity: dict | None = None) -> RowResult:
    """Run a single batch row (optimise or evaluate).

    ``on_candidate(iters)`` (optional) is forwarded to the optimiser for live
    within-column progress (a soil p-y optimise can take a while per column).
    ``silo`` (in) overrides the row's entered silo depth — used by the balance
    stage to re-analyse a pier at a trial depth.

    ``demand_basis`` / ``end_fixity`` come from :func:`_frame_basis` and make
    this bent be checked against its FRAME's period and end condition instead
    of its own stand-alone cantilever.  For a single-bent frame they reduce to
    exactly the stand-alone values, so passing them changes nothing there.
    """
    geometry, column, shaft, mults = _row_to_inputs(row, cfg, silo=silo)
    frame = str(row.get("frame", "") or "")
    deck_link = str(row.get("deck_link", "integral") or "integral")
    w_trans = float(row.get("weight_trans_kip", row["weight_long_kip"]))
    spectrum = cfg.design_spectrum.build()
    lle_spectrum = cfg.lle_spectrum.build() if cfg.lle_spectrum else None
    provisions = get_provisions(cfg.code)
    # The selected code's longitudinal limits govern; a user entry may only be
    # *stricter*.  Without this the provisions values were never read at all and
    # the GUI could silently accept rho_l below the code minimum.
    rho_l_min = max(cfg.rho_l_min, provisions.rho_l_min)
    rho_l_max = min(cfg.rho_l_max, provisions.rho_l_max)
    mu_d_limit = min(cfg.mu_d_limit, provisions.mu_d_limit_single)
    axial = float(row["axial_kip"])          # PER COLUMN dead-load share
    weight = float(row["weight_long_kip"])   # seismic suite: unchanged basis
    n_cols = max(1, int(row.get("n_columns", 1) or 1))
    col_spacing = float(row.get("col_spacing_ft", 0.0) or 0.0) * 12.0
    cap_fixity = str(row.get("cap_fixity", "fixed") or "fixed").strip().lower()
    # SDC 2.1 4.3.2 / SGS 4.9: a multi-column bent is allowed more displacement
    # ductility than a single-column one, having a redundant load path.
    if n_cols > 1:
        mu_d_limit = min(cfg.mu_d_limit, provisions.mu_d_limit_multi)
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
            weight_trans=w_trans, demand_basis=demand_basis,
            end_fixity=end_fixity, n_columns=n_cols, col_spacing=col_spacing,
            cap_fixity=cap_fixity,
            fixity_multipliers=mults, shaft_moment_basis=cfg.shaft_moment_basis,
            lle_spectrum=lle_spectrum, lle_mu_limit=cfg.lle_mu_limit,
            concrete_unit_weight=cfg.concrete_unit_weight,
            self_weight_mass_factor=cfg.self_weight_mass_factor,
            self_weight_in_axial=cfg.self_weight_in_axial,
            provisions=provisions, on_candidate=on_candidate, **soil_kw,
        )
        return RowResult(name, res.design, res.shaft, res.assessment, res.feasible,
                         True, res.log, frame=frame, silo=geometry.silo,
                         deck_link=deck_link, weight_trans=w_trans,
                         cap_fixity=cap_fixity)

    bent = evaluate_bent(
        n_cols, col_spacing, axial, cap_fixity=cap_fixity,
        column=column.section(), shaft=shaft.section(), geometry=geometry,
        spectrum=spectrum, weight=weight,
        weight_trans=w_trans, demand_basis=demand_basis, end_fixity=end_fixity,
        fixity_multipliers=mults, mu_d_limit=mu_d_limit,
        rho_l_min=rho_l_min, rho_l_max=rho_l_max,
        shaft_moment_basis=cfg.shaft_moment_basis,
        lle_spectrum=lle_spectrum, lle_mu_limit=cfg.lle_mu_limit,
        concrete_unit_weight=cfg.concrete_unit_weight,
        self_weight_mass_factor=cfg.self_weight_mass_factor,
        self_weight_in_axial=cfg.self_weight_in_axial,
        provisions=provisions, **soil_kw,
    )
    assessment = replace(bent.assessment, checks=bent.checks)
    return RowResult(name, column, shaft, assessment,
                     all(c.passed for c in bent.checks), False, list(bent.log),
                     frame=frame, silo=geometry.silo,
                     deck_link=deck_link, weight_trans=w_trans, bent=bent,
                     cap_fixity=cap_fixity)


def _criteria(cfg: GlobalConfig) -> BalanceCriteria:
    """Balance limits for this run — a user entry may only be *stricter*."""
    prov = get_provisions(cfg.code)
    return BalanceCriteria(
        k_ratio_min=max(cfg.balance_k_ratio_min, prov.balance_k_ratio_adjacent),
        k_ratio_any=max(cfg.balance_k_ratio_any, prov.balance_k_ratio_any),
        T_ratio_min=max(cfg.balance_T_ratio_min, prov.balance_T_ratio),
        mass_normalized=cfg.balance_mass_normalized,
        ref_stiffness=prov.ref_balanced_stiffness,
        ref_stiffness_any=prov.ref_balanced_stiffness_any,
        ref_geometry=prov.ref_balanced_geometry,
    )


def _build_bents(results: list[RowResult], order: dict[str, int],
                 cfg: GlobalConfig) -> list[BentStiffness]:
    """Bents that take part in the checks, in table row order.

    The longitudinal mass is the one the seismic run already used (entered
    weight plus the participating column self-weight).  The transverse mass is
    the entered transverse weight plus the SAME self-weight participation, so
    the two differ only by what the deck actually delivers to the bent.
    """
    out: list[BentStiffness] = []
    for rr in results:
        if not (in_frame(rr.frame) and rr.assessment.bounds):
            continue
        a = rr.assessment
        # Derive BOTH masses from the entered weight, never from the demand.
        # Reading m_long off ``bounds[0].demand.mass`` used to be equivalent,
        # but only while the demand was self-derived: once a bent is checked
        # against its FRAME's demand that field holds the frame's mass, which
        # would feed back in here and compound every pass.
        m_long = (a.weight_entered
                  + cfg.self_weight_mass_factor * a.W_self) / G_IN_S2
        m_trans = (rr.weight_trans
                   + cfg.self_weight_mass_factor * a.W_self) / G_IN_S2
        out.append(bent_stiffness(
            rr.name, rr.frame, order[rr.name], a,
            Hcol=a.Hcol_entered, silo=rr.silo, n_columns=rr.n_columns,
            cap_fixity=rr.cap_fixity,
            mass_long=m_long, mass_trans=m_trans, deck_link=rr.deck_link,
            D_shaft=rr.shaft.D))
    return out


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

    # Frame demand: a bent in a continuous frame does not respond on its own
    # period.  Runs before the balance stage, which reads the resulting EI.
    frame_log: list[str] = []
    if results:
        frame_log = _frame_demand_stage(results, rows_by_name, order, cfg,
                                        on_candidate=on_candidate,
                                        on_status=balance_progress)

    # Short-column shear: a pier can be perfectly balanced with its neighbours
    # and still fail its own shear check, because Vo = n*Mo/H_free and a short
    # column simply develops too much of it.  Size those silos FIRST so the
    # balance search starts from depths that are already necessary.
    shear_floors: dict = {}
    shear_log: list[str] = []
    if results:
        shear_floors, shear_log = _shear_silo_floors(
            results, rows_by_name, order, cfg, on_candidate=on_candidate,
            on_status=balance_progress)

    balance = None
    if cfg.balance_check:
        balance = _balance_stage(results, rows_by_name, order, cfg,
                                 on_candidate=on_candidate,
                                 on_status=balance_progress,
                                 shear_floors=shear_floors)

    frame_checks: list[FrameCheck] = []
    if balance is not None:
        # Only worth reporting where the frame differs from the stand-alone
        # cantilever the seismic suite already ran, i.e. a continuous frame.
        frame_checks = check_all(balance,
                                 {rr.name: rr.assessment for rr in results},
                                 cfg.design_spectrum.build(),
                                 get_provisions(cfg.code),
                                 soil_bounds=(cfg.soil_stiff_factor,
                                              cfg.soil_soft_factor))
    if balance is not None and (frame_log or shear_log):
        balance.log[:0] = shear_log + frame_log

    final = {rr.name: rr for rr in results}      # post-balance designs
    summary = pd.DataFrame([
        s if isinstance(s, dict) else _summary_row(final[s], balance)
        for s in slots])
    return BatchOutcome(summary=summary, results=results, balance=balance,
                        frame_checks=frame_checks)


def _silo_ctx(bents: list[BentStiffness], results: dict[str, RowResult],
              cfg: GlobalConfig, silos: dict[str, float]):
    """Predicted stiffness/mass at trial silo depths, calibrated to the real run.

    The raw elastic cantilever over-predicts how much a silo softens a pier: in
    soil mode lengthening the column also re-derives the point of fixity from the
    p-y solve, which claws some stiffness back.  So each pier's prediction is
    scaled to reproduce the stiffness actually measured at its current depth.
    Mass grows slightly with the silo too (more column self-weight participating),
    which lowers kappa and raises T — ignoring it would make the predictor
    optimistic.
    """
    corr: dict[str, float] = {}
    for b in bents:
        rr = results[b.name]
        ke = stiffness_at_silo(Geometry(Hcol=b.Hcol, D_shaft=rr.shaft.D),
                               rr.assessment.EI_col, rr.assessment.EI_shaft,
                               rr.assessment.bounds[0].multiplier, b.silo)
        corr[b.name] = (b.k[0] / ke) if ke > 0 else 1.0

    def k_at(b: BentStiffness, silo: float, bound: int,
             direction: str = LONGITUDINAL) -> float:
        """PER BENT, to match ``BentStiffness.stiffness`` and ``m_at``.

        The columns of a bent act in parallel, so the bent is ``n`` times one
        column.  Leaving the factor off makes the predictor disagree with the
        rule it is planning against wherever two neighbours carry different
        column counts -- a 2-column bent beside a 3-column one looks balanced
        to the planner while the real check fails, so it never softens anyone.
        """
        rr = results[b.name]
        return max(b.n_columns, 1) * corr[b.name] * stiffness_at_silo(
            Geometry(Hcol=b.Hcol, D_shaft=rr.shaft.D),
            rr.assessment.EI_col, rr.assessment.EI_shaft,
            rr.assessment.bounds[bound].multiplier, silo,
            end_fixity=b.end_fixity(direction))

    def to_raw(b: BentStiffness, k_bent: float) -> float:
        """A bent-level ``k_at`` target back in the raw per-column units that
        :func:`required_silo` bisects on."""
        return k_bent / (max(b.n_columns, 1) * (corr[b.name] or 1.0))

    def m_at(b: BentStiffness, silo: float, direction: str) -> float:
        rr = results[b.name]
        dW = column_self_weight(rr.design.section().Ag, silo - b.silo,
                                cfg.concrete_unit_weight)
        return b.mass(direction) + cfg.self_weight_mass_factor * dW / G_IN_S2

    return k_at, m_at, to_raw


def _plan_silos(bents: list[BentStiffness], results: dict[str, RowResult],
                criteria: BalanceCriteria, cfg: GlobalConfig,
                floors: dict[str, float]) -> tuple[dict[str, float], list[str]]:
    """Pairwise repair: deepen one silo at a time until every rule holds.

    A silo only ever *softens*, so for each failing check there is exactly one
    pier to lengthen:

    * **balanced stiffness** (inside a continuous frame, adjacent or any-two) —
      the member with the larger ``kappa`` comes down to ``kappa_other / limit``;
    * **balanced frame geometry** (between adjacent frames) — the frame with the
      SHORTER period must soften, so its stiffest member comes down by enough to
      bring ``K_frame`` to ``M_frame * (2*pi / (L_T * T_other))**2``.  Softening
      any member lowers ``K_frame`` and so lengthens ``T_frame``.

    Sweeps until the depths stop moving.  This is Gauss-Seidel on a monotone
    system, so it converges to the same minimum the exact search finds — it just
    needs more passes, and unlike the exact search it copes with the any-two rule
    and with frames coupled through ``K_frame``.
    """
    cap = cfg.max_silo_ft * 12.0
    step = cfg.silo_step_ft * 12.0
    silos = {b.name: max(b.silo, floors.get(b.name, 0.0)) for b in bents}
    notes: list[str] = []
    k_at, m_at, to_raw = _silo_ctx(bents, results, cfg, silos)

    def soften_to(target_k: float, who: BentStiffness, other: str,
                  bound: int, rule: str, direction: str) -> bool:
        """Plan a silo bringing ``who`` down to ``target_k``.  True if it moved.

        ``direction`` selects the end condition, so an integral bent is bisected
        on its fixed-fixed curve longitudinally and its fixed-free curve
        transversely — the same member, two different stiffness laws.
        """
        rr = results[who.name]
        geom = Geometry(Hcol=who.Hcol, D_shaft=rr.shaft.D)
        h = required_silo(geom, rr.assessment.EI_col, rr.assessment.EI_shaft,
                          rr.assessment.bounds[bound].multiplier,
                          to_raw(who, target_k),
                          silo_min=silos[who.name], silo_max=cap,
                          end_fixity=who.end_fixity(direction))
        if h is None:                      # the cap binds — go as deep as allowed
            notes.append(
                f"{who.name}: {cfg.max_silo_ft:g} ft silo cap is reached and "
                f"{rule} against {other} still fails at the "
                f"{who.label(bound)} bound — a silo only softens, so stiffen "
                f"{other} (larger column), rebalance the tributary mass, or "
                f"raise the cap")
            if silos[who.name] < cap:
                silos[who.name] = cap
                return True
            return False
        h = quantise_silo(h, step, cap, floor=silos[who.name])
        if h > silos[who.name] + 1e-9:
            silos[who.name] = h
            return True
        return False

    for _sweep in range(8):
        moved = False
        for direction in DIRECTIONS:
            frames = frames_for(bents, direction)

            # --- balanced stiffness, inside a continuous frame ---
            for f in frames:
                if not f.continuous:
                    continue
                near = {(a.name, b.name) for a, b in zip(f.members, f.members[1:])}
                for bound in range(f.n_bounds):
                    for bi, bj in itertools.combinations(f.members, 2):
                        limit = (criteria.k_ratio_min
                                 if (bi.name, bj.name) in near
                                 else criteria.k_ratio_any)
                        ki = k_at(bi, silos[bi.name], bound, direction)
                        kj = k_at(bj, silos[bj.name], bound, direction)
                        mi = m_at(bi, silos[bi.name], direction)
                        mj = m_at(bj, silos[bj.name], direction)
                        if not (math.isfinite(ki) and math.isfinite(kj)):
                            continue
                        ai = ki / mi if criteria.mass_normalized else ki
                        aj = kj / mj if criteria.mass_normalized else kj
                        if min(ai, aj) <= 0 or min(ai, aj) / max(ai, aj) >= limit:
                            continue
                        stiff, soft = (bi, bj) if ai > aj else (bj, bi)
                        target = min(ai, aj) / limit
                        if criteria.mass_normalized:
                            target *= m_at(stiff, silos[stiff.name], direction)
                        moved |= soften_to(target, stiff, soft.name, bound,
                                           f"balanced stiffness ({direction})",
                                           direction)

            # --- balanced frame geometry, between adjacent frames ---
            for fi, fj in zip(frames, frames[1:]):
                for bound in range(min(fi.n_bounds, fj.n_bounds)):
                    Ki = sum(k_at(b, silos[b.name], bound, direction)
                             for b in fi.members)
                    Mi = sum(m_at(b, silos[b.name], direction) for b in fi.members)
                    Kj = sum(k_at(b, silos[b.name], bound, direction)
                             for b in fj.members)
                    Mj = sum(m_at(b, silos[b.name], direction) for b in fj.members)
                    if min(Ki, Kj) <= 0 or min(Mi, Mj) <= 0:
                        continue
                    Ti = 2.0 * math.pi * math.sqrt(Mi / Ki)
                    Tj = 2.0 * math.pi * math.sqrt(Mj / Kj)
                    if min(Ti, Tj) / max(Ti, Tj) >= criteria.T_ratio_min:
                        continue
                    quick, slow = (fi, fj) if Ti < Tj else (fj, fi)
                    Kq, Mq = ((Ki, Mi) if quick is fi else (Kj, Mj))
                    T_target = criteria.T_ratio_min * max(Ti, Tj)
                    K_target = Mq * (2.0 * math.pi / T_target) ** 2
                    # take the whole reduction out of the stiffest member
                    who = max(quick.members,
                              key=lambda b: k_at(b, silos[b.name], bound,
                                                 direction))
                    k_who = k_at(who, silos[who.name], bound, direction)
                    k_target = max(k_who - (Kq - K_target), 1e-6)
                    moved |= soften_to(k_target, who, slow.key, bound,
                                       f"balanced frame geometry ({direction})",
                                       direction)
        if not moved:
            break
    return silos, dedupe(notes)


def _all_frames_single(bents: list[BentStiffness]) -> bool:
    """True when no frame has two bents acting together, in either direction."""
    return all(not f.continuous
               for d in DIRECTIONS for f in frames_for(bents, d))


def _plan_silos_min(bents: list[BentStiffness], results: dict[str, RowResult],
                    criteria: BalanceCriteria, cfg: GlobalConfig,
                    floors: dict[str, float]) -> tuple[dict[str, float], list[str]]:
    """Minimum-total-silo plan — exact on the buildable grid, where it applies.

    With every frame a single bent (a run of simply supported spans) the only
    rule left is balanced frame geometry between neighbours, so the bridge is a
    chain and :func:`~seismic_column.balance.dp_min_silo` gives the true optimum.

    A continuous frame breaks that: the any-two-bents rule makes it all-pairs,
    and the period rule couples whole frames through ``K_frame``.  Neither fits a
    neighbour-only chain, so :func:`_balance_stage` falls back to the pairwise
    repair rather than claim an optimality this no longer has.
    """
    cap = cfg.max_silo_ft * 12.0
    step = cfg.silo_step_ft * 12.0
    # States start at the ENTERED floor, not at the current trial depth: a
    # minimiser must be free to take a silo back off once a better-calibrated
    # pass shows less is needed.  (The greedy repair is deliberately monotone;
    # this is not.)
    states = {b.name: silo_states(floors.get(b.name, 0.0), cap, step)
              for b in bents}
    k_at, m_at, _to_raw = _silo_ctx(bents, results, cfg,
                                    {b.name: b.silo for b in bents})

    def feasible(bi: BentStiffness, si: float,
                 bj: BentStiffness, sj: float) -> bool:
        """Single-bent frames, so only the period rule couples the two — and it
        must hold in BOTH directions and at every bound."""
        for direction in DIRECTIONS:
            for bound in range(min(len(bi.k), len(bj.k))):
                ki = k_at(bi, si, bound, direction)
                kj = k_at(bj, sj, bound, direction)
                mi, mj = m_at(bi, si, direction), m_at(bj, sj, direction)
                if min(ki, kj) <= 0 or min(mi, mj) <= 0:
                    return False
                Ti, Tj = math.sqrt(mi / ki), math.sqrt(mj / kj)   # 2*pi cancels
                if min(Ti, Tj) / max(Ti, Tj) < criteria.T_ratio_min:
                    return False
        return True

    ordered = [b for b in sorted(bents, key=lambda b: b.order)
               if in_frame(b.frame)]
    plan = dp_min_silo(bents, states, feasible, chain=[ordered])
    return plan.silos, plan.notes


def _frame_basis(results: list[RowResult], order: dict[str, int],
                 cfg: GlobalConfig) -> dict[str, tuple[dict, dict]]:
    """Per pier, the (K, W) its demand comes from and its end condition.

    A bent inside a continuous frame does not respond on its own period.  The
    deck ties the members together, so they share a displacement and the demand
    is the FRAME's::

        K_frame = sum k_i   (each member at ITS end condition in this direction)
        W_frame = sum w_i

    Returns ``{pier: (demand_basis, end_fixity)}`` ready for :func:`run_row`.
    A single-bent frame yields exactly that bent's own ``(k, w)``, so this is a
    no-op for the simply supported reaches — which is what makes it safe to
    apply everywhere rather than special-casing continuous frames.
    """
    bents = _build_bents(results, order, cfg)
    out: dict[str, tuple[dict, dict]] = {}
    _softness: dict[tuple[str, str], float] = {}
    for direction in DIRECTIONS:
        for f in frames_for(bents, direction):
            n = f.n_bounds
            if n < 1:            # nothing solved for this frame yet
                continue
            # Keyed by bound LABEL, not index: bent_stiffness collapses
            # duplicate bounds, so a positional list would misalign with the
            # full bound set evaluate_column solves and silently leave the
            # extra bounds on their stand-alone demand.
            lbls = f.members[0].bound_labels
            basis = {lbls[i]: (f.K(i), f.M() * G_IN_S2) for i in range(n)}
            # A boundary bent belongs to two frames.  Take the one that governs
            # it -- the LONGER period, since displacement capacity is what these
            # bents fail on and spectral displacement grows with T.  (Shear is
            # capacity-based, Vo = n*Mo/H_free, so it does not enter here.)
            soft = f.M() / f.K(0) if f.K(0) > 0 else float("inf")
            for b in f.members:
                d_basis, d_fix = out.setdefault(b.name, ({}, {}))
                if direction in d_basis and soft <= _softness.get(
                        (b.name, direction), -1.0):
                    continue
                _softness[(b.name, direction)] = soft
                d_basis[direction] = basis
                d_fix[direction] = b.end_fixity(direction)
    return out


def _frame_demand_stage(results: list[RowResult],
                        rows_by_name: dict[str, pd.Series],
                        order: dict[str, int], cfg: GlobalConfig,
                        on_candidate=None, on_status=None,
                        max_passes: int = 6) -> list[str]:
    """Re-run every row against its FRAME's demand, to a fixed point.

    Sizing one member changes ``K_frame`` and so the demand on every member of
    that frame, which is why this iterates rather than running once.

    It converges because growth helps twice over: a bigger member raises ``k``
    roughly with ``D^4`` through EI while raising mass only with ``D^2``, so
    ``K_frame`` rises, ``T_frame`` falls and the demand drops for **every**
    member — at the same time as that member's own capacity rises.

    The one way it could cycle is a member SHRINKING on a later pass (the
    optimiser starts from minimum reinforcement each call), undoing the frame
    stiffening.  Designs are therefore held monotone: a pass may grow a column
    but never shrink it below what an earlier pass established.
    """
    log: list[str] = []
    by_name = {rr.name: rr for rr in results}
    best_D = {rr.name: rr.design.D for rr in results}
    prev_sig = None
    prev_basis: dict = {}

    # Only bents that SHARE a frame with someone need re-running: a single-bent
    # frame's (K, W) is that bent's own, so the stage would reproduce the
    # stand-alone result it already has.  On a real bridge that skips most of
    # the deck — 13 of 18 piers here — and this stage re-runs the optimiser, so
    # the saving is the difference between minutes and tens of minutes.
    shared: set[str] = set()
    for direction in DIRECTIONS:
        for f in frames_for(_build_bents(results, order, cfg), direction):
            if len(f.members) > 1:
                shared.update(f.names)
    if not shared:
        return log

    for p in range(1, max_passes + 1):
        basis = _frame_basis(results, order, cfg)
        if not basis:
            return log
        if on_status is not None:
            on_status(f"Frame demand: pass {p} ({len(shared)} in shared frames)")
        changed: list[str] = []
        for i, rr in enumerate(results):
            nb = basis.get(rr.name)
            if nb is None or rr.name not in shared:
                continue
            # nothing moved for this pier since the last pass — its design
            # already satisfies this demand, so re-running is wasted work
            if prev_basis.get(rr.name) == nb[0]:
                continue
            row = rows_by_name[rr.name].copy()
            # monotone guard: never let a later pass undercut an earlier one
            row["Dcol_in"] = max(float(row["Dcol_in"]), best_D[rr.name])
            new = run_row(row, cfg, on_candidate=on_candidate, silo=rr.silo,
                          demand_basis=nb[0], end_fixity=nb[1])
            if new.design.D != rr.design.D:
                changed.append(f"{rr.name} {rr.design.D:.0f}→{new.design.D:.0f} in")
            best_D[rr.name] = max(best_D[rr.name], new.design.D)
            results[i] = new
            by_name[rr.name] = new
        prev_basis = {n: b[0] for n, b in basis.items()}
        sig = tuple(sorted((r.name, r.design.D, r.design.n_bars,
                            r.design.long_bar_no) for r in results))
        if sig == prev_sig:
            log.append(f"Frame demand converged after {p} pass(es).")
            return log
        prev_sig = sig
        if changed:
            log.append(f"Frame demand pass {p} resized: {', '.join(changed)}.")
    log.append(f"Frame demand did NOT settle within {max_passes} passes — the "
               f"designs reported are from the last pass, not a converged "
               f"fixed point. Treat them as provisional.")
    return log


def shear_silo_required(rr) -> float:
    """Extra silo a pier needs to bring its COLUMN SHEAR within capacity, in.

    A plastic hinge develops ``Vo = n*Mo/H_free`` -- one hinge for a cantilever,
    two for a fixed-fixed member.  ``Vo`` is therefore inversely proportional to
    the free length, and a SHORT column develops enormous shear when it hinges.
    Transverse steel cannot fix that: more spiral raises the confined strength,
    which raises Mp, which raises Vo again.  The lever is the only thing that
    actually moves the demand.

    So the silo is the lever, and the required length follows directly from the
    demand/capacity ratio::

        Vo ~ 1/H_free   =>   H_required = H_free * (Vo / phiVn)

    The balance search never finds this, because it sizes silos for stiffness
    RATIOS between piers, not for a pier's own capacity -- a short pier can be
    perfectly balanced with its neighbours and still fail its own shear check.

    Returns 0 when the shear check already passes.  The 10 % margin covers Mo
    drifting a little with the plastic hinge length as the column lengthens;
    the caller iterates, so it only has to be close.
    """
    ck = next((c for c in rr.assessment.checks if c.name == "Column shear"), None)
    if ck is None or ck.capacity <= 0.0:
        return 0.0
    dc = ck.demand / ck.capacity
    if dc <= 1.0:
        return 0.0
    return max(0.0, rr.assessment.H_free * (dc - 1.0) * 1.10)


def _shear_silo_floors(results: list[RowResult], rows_by_name: dict,
                       order: dict, cfg: GlobalConfig, on_candidate=None,
                       on_status=None, max_passes: int = 4) -> tuple[dict, list]:
    """Silo floors that make every pier's own column-shear check pass.

    Run BEFORE the balance stage so its search starts from a set of depths that
    are already structurally necessary, rather than discovering later that a
    balanced bridge still has piers failing on shear.  Lengthening a column
    changes Lp, Mo and the demand, so this iterates.
    """
    floors: dict[str, float] = {}
    log: list[str] = []
    for p in range(1, max_passes + 1):
        need = {}
        for rr in results:
            extra = shear_silo_required(rr)
            if extra > 1e-6:
                need[rr.name] = floors.get(rr.name, rr.silo) + extra
        if not need:
            break
        if on_status is not None:
            on_status(f"Short-column shear: pass {p} ({len(need)} pier(s))")
        for i, rr in enumerate(results):
            if rr.name not in need:
                continue
            floors[rr.name] = need[rr.name]
            try:
                # Keep the FRAME basis on the re-run.  Without it this stage
                # silently reverts the pier to a stand-alone cantilever on its
                # own period -- undoing the end condition and demand the frame
                # stage just established, and changing the very shear it is
                # trying to size for.
                nb = _frame_basis(results, order, cfg).get(rr.name)
                results[i] = run_row(rows_by_name[rr.name], cfg,
                                     on_candidate=on_candidate,
                                     silo=floors[rr.name],
                                     demand_basis=nb[0] if nb else None,
                                     end_fixity=nb[1] if nb else None)
            except Exception as exc:
                log.append(f"{rr.name}: could not re-analyse at a "
                           f"{floors[rr.name]/12:.1f} ft silo ({exc}).")
                floors.pop(rr.name, None)
    if floors:
        log.insert(0, (
            "Column shear drove a silo on "
            + ", ".join(f"{n} {v/12:.0f} ft" for n, v in sorted(floors.items()))
            + ". A short column develops Vo = n*Mo/H_free when it hinges, so "
              "the free length is the only lever that lowers it — transverse "
              "steel raises Mp and with it Vo. These are floors; the balance "
              "search may deepen them but never reduces them."))
    return floors, log


def _balance_stage(results: list[RowResult], rows_by_name: dict[str, pd.Series],
                   order: dict[str, int], cfg: GlobalConfig,
                   on_candidate=None, on_status=None,
                   shear_floors: dict | None = None) -> BalanceResult:
    """Check adjacent-pier balance and, if enabled, auto-size column silos.

    Returns the final :class:`~seismic_column.balance.BalanceResult`; ``results``
    is mutated in place so its entries always describe the design that was
    finally checked.
    """
    criteria = _criteria(cfg)
    bents = _build_bents(results, order, cfg)
    checks = balance_checks(bents, criteria)
    log: list[str] = []

    if len(bents) < 2:
        log.append("Fewer than two piers take part in the balance checks "
                   "(check the 'frame' column) — nothing to compare.")
    frames = {d: frames_for(bents, d) for d in DIRECTIONS}
    result = BalanceResult(bents=bents, checks=checks, criteria=criteria,
                           frames=frames, log=log, converged=True)
    if result.passed or not cfg.balance_auto_silo or len(bents) < 2:
        if result.passed and len(bents) >= 2:
            log.append("All adjacent pairs comply as designed — no silo needed.")
        elif not cfg.balance_auto_silo and not result.passed:
            log.append("Auto-silo is off: reporting the shortfall without "
                       "changing any design.")
        return result

    by_name = {rr.name: rr for rr in results}
    floors = {rr.name: max(
        float(rows_by_name[rr.name].get("silo_ft", 0.0) or 0.0) * 12.0,
        (shear_floors or {}).get(rr.name, 0.0))
        for rr in results}

    minimise = cfg.balance_strategy == "min_silo"
    if minimise and not _all_frames_single(bents):
        minimise = False
        log.append(
            "Silo strategy: fell back to pairwise repair. The exact search "
            "solves a chain of neighbour-only constraints, but a continuous "
            "frame is present — the any-two-bents rule makes it all-pairs and "
            "the period rule couples whole frames through K_frame, neither of "
            "which is a chain. The repair still converges; it just cannot claim "
            "to be provably minimal here.")
    planner = _plan_silos_min if minimise else _plan_silos
    if minimise:
        log.append("Silo strategy: minimum total depth (exact on the buildable "
                   f"{cfg.silo_step_ft:g} ft grid), verified against a real "
                   "analysis each pass.")
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
                # keep the FRAME demand basis on the re-run: a silo changes
                # H_free and so k, hence K_frame, and dropping the basis here
                # would quietly revert this pier to its stand-alone demand
                _nb = _frame_basis(results, order, cfg).get(name)
                rr = run_row(rows_by_name[name], cfg, on_candidate=row_cb,
                             silo=silos[name],
                             demand_basis=_nb[0] if _nb else None,
                             end_fixity=_nb[1] if _nb else None)
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

        bents = _build_bents(results, order, cfg)
        checks = balance_checks(bents, criteria)
        result.bents, result.checks = bents, checks
        result.frames = {d: frames_for(bents, d) for d in DIRECTIONS}
        if result.passed:
            total = sum(r.silo for r in results)
            if best is None or total < best[0] - 1e-9:
                best = (total, list(results), list(bents), list(checks),
                        dict(result.frames))
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
            result.bents, result.checks, result.frames = best[2], best[3], best[4]
            by_name = {rr.name: rr for rr in results}
            log.append(f"Kept the cheapest feasible plan found: "
                       f"{best[0] / 12:g} ft of silo.")
        result.converged = True

    # The search has now done everything it can.  A BETWEEN-FRAME geometry pair
    # still short at this point is not a design failure to be beaten out with a
    # deeper silo -- the balance rules exist to let a bridge AVOID a more
    # rigorous analysis, and where they cannot be met the code's own route is
    # nonlinear time-history (SDC 7.1.3 / SGS 4.1.3).  So it is referred there.
    #
    # WITHIN-FRAME stiffness is deliberately NOT referred: that rule governs how
    # the frame distributes demand between its own members, and a time-history
    # run does not excuse it.
    unresolved_geo = [c for c in result.checks
                      if not c.passed and c.name == GEOMETRY_CHECK]
    if unresolved_geo:
        for c in unresolved_geo:
            c.tha_required = True
        pairs = sorted({f"{c.pair[0]}–{c.pair[1]}" for c in unresolved_geo})
        log.append(
            f"TIME-HISTORY REQUIRED — balanced frame geometry could not be met "
            f"by practical means at {len(unresolved_geo)} pair(s) "
            f"({', '.join(pairs)}). The silo search pursued it and stopped; "
            f"closing it further would need silo depths or diameter changes "
            f"beyond what the search could reach. Per SDC 7.1.3 / SGS 4.1.3 "
            f"this is cleared by nonlinear time-history analysis, not by "
            f"forcing the ratio. The WITHIN-frame stiffness rules are still "
            f"enforced and are reported separately.")

    if not result.passed:
        for c in result.failed:
            log.append(f"UNRESOLVED — {c.label}: ratio {c.ratio:.3f} "
                       f"< {c.limit:.2f} ({c.note})")
    for c in result.needs_tha:
        log.append(f"THA — {c.label}: ratio {c.ratio:.3f} < {c.limit:.2f} "
                   f"({c.note})")
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
        # stiffness is keyed on the pier; the geometry rule is keyed on the
        # FRAME the pier sits in, which differs by direction
        keys = balance.frames_of(rr.name) | {rr.name}
        def _worst(check_name, direction=None):
            vals = [c.ratio for c in balance.checks
                    if c.name == check_name and keys & set(c.pair)
                    and (direction is None or c.direction == direction)
                    and not math.isnan(c.ratio)]
            return round(min(vals), 3) if vals else None
        row.update({
            "frame": rr.frame,
            "deck_link": rr.deck_link,
            "T_long_s": round(a.bounds[0].demand.period, 3),
            "bal_k_ratio": min([v for v in (_worst(STIFFNESS_CHECK),
                                            _worst(STIFFNESS_ANY_CHECK))
                                if v is not None], default=None),
            "bal_T_long": _worst(GEOMETRY_CHECK, LONGITUDINAL),
            "bal_T_trans": _worst(GEOMETRY_CHECK, TRANSVERSE),
            "balanced": ("-" if not balance.touches(rr.name)
                         else ("PASS" if balance.pier_passed(rr.name) else "FAIL")),
        })
    return row
