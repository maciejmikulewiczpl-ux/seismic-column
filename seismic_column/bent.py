"""A BENT of one or more columns, and the transverse push/pull between them.

A single-column bent is a cantilever in both directions and this module is a
thin pass-through.  A multi-column bent is different in three ways that all
matter, and only the first is obvious:

**1. Transversely it is a portal frame.**  Pushing it overturns the bent, and
the overturning is resisted by an axial COUPLE between the columns — the
windward column is unloaded (or put into net tension), the leeward one is
compressed.  Axial changes the moment-curvature, so ``Mp``, ``Vo``, ``Δy``,
``Δc`` and even the depth to fixity all differ between columns of the SAME
bent, from the same section.

**2. The cap restrains the column heads.**  A multi-column bent with an integral
cap is fixed-fixed transversely, not the cantilever a single-column bent is, so
each column develops the two-hinge mechanism shear ``2*Mp/H``.  That inverts the
usual rule that transverse is always fixed-free.

**3. Longitudinally there is no couple at all.**  The columns stand at one
longitudinal station, so pushing along the bridge puts them all in the same
curvature.  They act as ``n`` identical members in parallel at the dead-load
axial, and the end condition is whatever ``deck_link`` says, exactly as before.

Because of (3) the two directions do not share an axial, so they cannot share a
run: the transverse checks come from the two extreme positions at
``P_dead ± ΔP``, and the longitudinal ones from a further run at ``P_dead``.
Taking longitudinal off a ±ΔP run would be wrong in both directions at once.

``P_dead`` is the bent reaction shared between the columns — the tributary
weights in the table are per bent, and the axial is read the same way.

**Not covered.**  The cap beam itself — its flexure, its shear, and the
column-to-cap joint (SDC 7.4).  A multi-column bent needs all three and this
module checks none of them; the report says so rather than leaving it implied.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from .io_schema import LONGITUDINAL, TRANSVERSE
from .sdc_capacity import (Check, ColumnAssessment, evaluate_column)


@dataclass
class ColumnPosition:
    """One column of a bent, and the axial the transverse couple gives it."""

    index: int
    x: float                    # offset from the bent centreline, in (+ = leeward)
    delta_P: float              # axial from the overturning couple, kip (+ = compr.)
    axial: float                # dead-load share + delta_P, kip
    assessment: ColumnAssessment

    @property
    def label(self) -> str:
        side = "leeward" if self.x > 0 else ("windward" if self.x < 0 else "centre")
        return f"col {self.index + 1} ({side})"

    @property
    def net_tension(self) -> bool:
        """Net tension kills the concrete shear term — worth flagging loudly."""
        return self.axial <= 0.0


@dataclass
class BentAssessment:
    """One bent: its columns, the couple between them, and the envelope."""

    n_columns: int
    spacing: float                       # centre-to-centre, in
    positions: list[ColumnPosition]
    checks: list[Check]                  # worst of every named check, any position
    governing: ColumnPosition
    delta_P: float = 0.0                 # the extreme couple axial, kip
    M_overturn: float = 0.0              # bent overturning at the fixity level
    V_bent: float = 0.0                  # overstrength shear the bent develops
    iterations: int = 0
    converged: bool = True
    log: list[str] = field(default_factory=list)

    @property
    def multi(self) -> bool:
        return self.n_columns > 1

    @property
    def assessment(self) -> ColumnAssessment:
        """The governing column — what a single-column caller expects."""
        return self.governing.assessment

    @property
    def k_total(self) -> dict:
        """Bent stiffness per direction per bound label: the columns add."""
        out: dict = {}
        for d in (LONGITUDINAL, TRANSVERSE):
            acc: dict = {}
            for p in self.positions:
                dr = p.assessment.directions.get(d)
                if dr is None:
                    continue
                for b in dr.bounds:
                    lbl = b.soil_label or f"{b.multiplier:.2g}·D_shaft"
                    acc[lbl] = acc.get(lbl, 0.0) + b.stiffness
            out[d] = acc
        return out


def offsets(n: int, spacing: float) -> list[float]:
    """Column offsets from the centreline, evenly spaced and centred."""
    return [(i - (n - 1) / 2.0) * spacing for i in range(n)]


def couple_axials(n: int, spacing: float, M_ot: float) -> list[float]:
    """Distribute an overturning moment as an axial couple across the columns.

    Linear (plane-sections) distribution: ``ΔPᵢ = M_ot·xᵢ / Σxⱼ²``.  For two
    columns that reduces to the familiar ``M_ot/s``.  The result always sums to
    zero — an overturning couple adds no net axial to the bent — and reproduces
    ``M_ot`` about the centreline, which is asserted in the tests rather than
    assumed here.
    """
    xs = offsets(n, spacing)
    denom = sum(x * x for x in xs)
    if denom <= 0.0:
        return [0.0] * n
    return [M_ot * x / denom for x in xs]


def evaluate_bent(n_columns: int, spacing: float, axial: float,
                  end_fixity: dict | None = None, demand_basis: dict | None = None,
                  cap_fixity: str = "fixed",
                  max_passes: int = 8, tol: float = 0.01,
                  **kw) -> BentAssessment:
    """Assess a bent of ``n_columns`` identical columns.

    ``axial`` is the BENT's dead-load reaction — the same basis as the tributary
    weights, which are also per bent — and is shared equally between the columns.
    ``spacing`` is centre-to-centre in inches.  Remaining keyword arguments go
    straight to :func:`~seismic_column.sdc_capacity.evaluate_column`.

    Only the two OUTERMOST columns are analysed at their own axial.  ΔP is linear
    in ``x``, so they take the largest swing in either direction and one of them
    always governs; the interior columns are represented by the dead-load run.
    A bent of any width therefore costs three analyses, not ``n + 1``.

    With ``n_columns == 1`` this is a pass-through and returns exactly what
    ``evaluate_column`` would — no couple, no extra runs, no behaviour change.

    Above one it iterates, because the push/pull is circular: ``Mp`` depends on
    the axial, the bent's overstrength shear depends on ``Mp``, and the couple
    depends on that shear.  It settles quickly — the M-P curve is shallow near
    service axial, so what the leeward column gains the windward one roughly
    gives back and ``V_bent`` barely moves.
    """
    ef = dict(end_fixity or {})
    # ``axial`` is the BENT's dead-load reaction, matching the tributary weights,
    # which are also per bent.  The columns share it.
    P_dead = axial / max(n_columns, 1)
    if n_columns <= 1:
        a = evaluate_column(axial=P_dead, end_fixity=ef or None,
                            demand_basis=demand_basis, **kw)
        pos = ColumnPosition(0, 0.0, 0.0, P_dead, a)
        return BentAssessment(1, spacing, [pos], list(a.checks), pos)

    # A PINNED cap develops no frame action at all: hinge at the base only, so
    # V*H = sum(Mo) and the base moments are already sum(Mo) -- the couple is
    # exactly zero and each column is an independent cantilever.  So there is
    # nothing to iterate and nothing to envelope beyond the single run.
    if cap_fixity == "pinned":
        a = evaluate_column(axial=P_dead, end_fixity=ef or None,
                            demand_basis=demand_basis, **kw)
        xs0 = offsets(n_columns, spacing)
        poss = [ColumnPosition(i, x, 0.0, P_dead, a)
                for i, x in enumerate(xs0)]
        return BentAssessment(
            n_columns, spacing, poss, list(a.checks), poss[0],
            converged=True, iterations=0,
            log=[f"{n_columns} columns at {spacing/12:.1f} ft with a PINNED "
                 f"cap: each column is an independent cantilever, so there is "
                 f"no push/pull couple (V·H = ΣMo is taken entirely by the base "
                 f"moments) and the transverse head stays fixed-FREE. Vo is "
                 f"Mo/H, half the monolithic-cap value."])

    # A monolithic cap holds the column heads against rotation, so transversely
    # the bent is a portal frame -- fixed-fixed -- not the cantilever a single
    # column is.
    ef[TRANSVERSE] = "fixed"
    xs = offsets(n_columns, spacing)
    log: list[str] = []
    dPs = [0.0] * n_columns
    M_ot = V_bent = 0.0
    it = 0
    converged = False
    # Only the OUTERMOST columns are analysed.  ΔP is linear in x, so they carry
    # the largest swing either way and one of them always governs; the interior
    # columns sit between them and are represented by the dead-load run.  That
    # keeps the cost at three analyses regardless of how wide the bent is.
    ends = (0, n_columns - 1)

    def _run_at(dP: float, dead):
        return dead if dP == 0.0 and dead is not None else evaluate_column(
            axial=P_dead + dP, end_fixity=ef, demand_basis=demand_basis, **kw)

    def _V_bent(a_lo, a_hi, a_mid) -> float:
        """Overstrength shear of the whole bent.

        The two extremes are analysed; the ``n-2`` interior columns are taken at
        the dead-load axial.  What the leeward column gains the windward one
        roughly gives back, so the sum is barely sensitive to how the middle is
        treated — on a two-column bent this reproduces the exact sum to ~1%.
        """
        v = (a_lo.directions[TRANSVERSE].Vo + a_hi.directions[TRANSVERSE].Vo)
        if n_columns > 2:
            v += (n_columns - 2) * a_mid.directions[TRANSVERSE].Vo
        return v

    def _M_ot(a_lo, a_hi, a_mid) -> float:
        m = a_lo.Mo + a_hi.Mo
        if n_columns > 2:
            m += (n_columns - 2) * a_mid.Mo
        return m

    mid = evaluate_column(axial=P_dead, end_fixity=ef,
                          demand_basis=demand_basis, **kw)
    for it in range(1, max_passes + 1):
        lo = _run_at(dPs[ends[0]], mid)
        hi = _run_at(dPs[ends[1]], mid)
        # How much of the overturning the axial COUPLE has to take.  Cut the
        # frame at the top of shaft, where the mechanism hinges are and where Mo
        # is defined, and take moments about the centreline:
        #
        #     V_bent * H_free = sum(dP_i * x_i) + sum(Mo_i)
        #
        # and since V_bent = sum(2*Mo_i / H_free), the left side is 2*sum(Mo_i),
        # so the couple carries
        #
        #     sum(dP_i * x_i) = sum(Mo_i)
        #
        # The column base moments take the other half.  Charging the couple with
        # the whole of V*Le -- the full height to the point of fixity, base
        # moments ignored -- overstates it several times over and can invent net
        # tension that is not there.  Note this is independent of Df.
        V_bent = _V_bent(lo, hi, mid)
        M_ot = _M_ot(lo, hi, mid)
        new_dPs = couple_axials(n_columns, spacing, M_ot)
        shift = max(abs(a - b) for a, b in zip(new_dPs, dPs))
        dPs = new_dPs
        if shift <= tol * max(abs(max(dPs, key=abs)), 1.0):
            converged = True
            break

    lo = _run_at(dPs[ends[0]], mid)
    hi = _run_at(dPs[ends[1]], mid)
    V_bent = _V_bent(lo, hi, mid)
    # ``M_overturn`` stays the value that GENERATED the reported dP, so the
    # couple reconciles exactly: sum(dP*x) == M_overturn.  A fixed point always
    # has one step of lag somewhere, and it is better placed against sum(Mo) of
    # the final runs -- where convergence bounds it -- than against the couple,
    # where it would make the reported table fail its own statics check.
    # Longitudinally there is NO couple: the columns are at one station, so they
    # all sit at the dead-load axial.  Its own run, because taking it off a
    # +/-dP run would be wrong in both directions at once.
    ef_lon = dict(ef)
    ef_lon.pop(TRANSVERSE, None)
    lon = evaluate_column(axial=P_dead, end_fixity=ef_lon or None,
                          demand_basis=demand_basis, **kw)
    # The outermost columns are analysed at their own axial; the interior ones
    # are represented by the dead-load run, since their swing is smaller and one
    # of the extremes always governs.
    runs = [lo if i == ends[0] else (hi if i == ends[1] else mid)
            for i in range(n_columns)]

    positions = [ColumnPosition(i, x, dP, P_dead + dP, a)
                 for i, (x, dP, a) in enumerate(zip(xs, dPs, runs))]
    per_pos: list[tuple[ColumnPosition, list[Check]]] = [
        (p, p.assessment.directions[TRANSVERSE].checks) for p in positions]
    # the longitudinal side is the same for every column, so attribute it to the
    # centre and let the envelope decide
    lon_pos = ColumnPosition(-1, 0.0, 0.0, P_dead, lon)
    per_pos.append((lon_pos, lon.directions[LONGITUDINAL].checks))
    checks = _envelope(per_pos)

    worst = min(
        positions,
        key=lambda p: min(
            (b.delta_c / b.demand.disp_demand
             for b in p.assessment.directions[TRANSVERSE].bounds
             if b.demand.disp_demand > 0), default=float("inf")))
    log.append(
        f"{n_columns} columns at {spacing/12:.1f} ft: transverse push/pull "
        f"±{max(abs(d) for d in dPs):.0f} kip on a per-column dead-load axial "
        f"of {P_dead:.0f} kip ({axial:.0f} kip on the bent, shared "
        f"{n_columns} ways) — "
        f"{'converged' if converged else 'NOT converged'} in {it} pass(es).")
    if n_columns > 2:
        log.append(
            f"Only the two outermost columns are analysed at their own axial; "
            f"ΔP is linear in x so they carry the largest swing and one of them "
            f"governs. The {n_columns - 2} interior column(s) are represented by "
            f"the dead-load run.")
    if any(p.net_tension for p in positions):
        names = ", ".join(p.label for p in positions if p.net_tension)
        log.append(
            f"NET TENSION at {names} — the concrete shear term vc goes to zero "
            f"there (SDC 5.3.7.2 / SGS 8.6.2-4), so that column relies on its "
            f"transverse steel alone.")
    if not converged:
        log.append(
            f"The push/pull did NOT settle within {max_passes} passes; the "
            f"axials reported are from the last one, not a fixed point.")

    return BentAssessment(n_columns, spacing, positions, checks, worst,
                          delta_P=max(abs(d) for d in dPs), M_overturn=M_ot,
                          V_bent=V_bent, iterations=it, converged=converged,
                          log=log)


def _envelope(per_position: list[tuple[ColumnPosition, list[Check]]]) -> list[Check]:
    """Worst of each named check across the column positions.

    Same rule as the two-direction envelope in :mod:`sdc_capacity`: a failing
    check always beats a passing one, otherwise the larger demand/capacity
    ratio wins.  The governing position is named only where the positions
    actually disagree, so a single-column bent stays quiet.
    """
    merged: dict[str, tuple[ColumnPosition, Check]] = {}
    for pos, checks in per_position:
        for c in checks:
            prev = merged.get(c.name)
            if prev is None:
                merged[c.name] = (pos, c)
                continue
            _, pc = prev
            worse = (pc.passed and not c.passed) or (
                pc.passed == c.passed and c.ratio > pc.ratio)
            if worse:
                merged[c.name] = (pos, c)

    out: list[Check] = []
    for pos, c in merged.values():
        differs = any(
            abs(x.ratio - c.ratio) > 1e-9 or x.passed != c.passed
            for p2, lst in per_position if p2 is not pos
            for x in lst if x.name == c.name)
        out.append(replace(c, note=(f"{c.note}  [{pos.label} governs]".strip()
                                    if differs else c.note)))
    return out
