"""Balanced stiffness and balanced frame geometry between adjacent piers.

The columns in the batch table support **simply supported spans in series**, so
adjacent piers interact even though each one is analysed as a stand-alone
single-column bent.  Two code rules govern that interaction:

*Balanced stiffness* — Caltrans SDC 2.1 §7.1.2 Table 7.1.2-1, AASHTO SGS 3rd Ed.
§4.1.2 Eq. 4.1.2-3/-4.  For **adjacent** bents::

    min(kappa_i, kappa_j) / max(kappa_i, kappa_j) >= 0.75

where ``kappa = k_e / m`` (the stiffness-to-mass ratio Caltrans always uses, and
the AASHTO variable-width form Eq. 4.1.2-4) or ``kappa = k_e`` (the AASHTO
constant-width form Eq. 4.1.2-3), selected by ``BalanceCriteria.mass_normalized``.

*Balanced frame geometry* — Caltrans SDC 2.1 §7.1.3, AASHTO SGS 3rd Ed. §4.1.3::

    min(T_i, T_j) / max(T_i, T_j) >= 0.70

Writing both as ``min/max`` makes Caltrans' two-sided limits (0.75…1.33 and
0.7…1.43) and AASHTO's one-sided ones the same test, and makes the check
independent of which pier is called *i*.

Scope, as specified for this project: **adjacent pairs only**.  The any-two-bents
0.50 rule (SDC Eq. 7.1.2-1 / SGS Eq. 4.1.2-1) is not applied — with an expansion
joint at every pier, each pier is arguably its own frame, and the piers in series
are treated as one frame of bents purely so the adjacent-bent rule bites.

Note the identity ``T = 2*pi*sqrt(m/k)``, so ``(T_i/T_j)^2 = kappa_j/kappa_i``:
under mass normalisation a stiffness ratio of 0.75 gives a period ratio of
``sqrt(0.75) = 0.866 >= 0.70``, i.e. the balanced-stiffness rule *implies* the
balanced-geometry rule.  Both are still evaluated — they are separately named
clauses, and they decouple under the constant-width (non-normalised) form.

The tuning lever is the **column silo** (isolation casing), which lengthens the
free column and so softens a stiff pier — Caltrans C7.1.2 / SGS §4.1.4.
:func:`required_silo` sizes it from the elastic two-segment cantilever, which
costs nothing (no moment-curvature, no p-y), so the batch orchestration can
predict a silo depth before paying for a full re-analysis at that depth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from .geometry import Geometry

STIFFNESS_CHECK = "Balanced stiffness (adjacent)"
GEOMETRY_CHECK = "Balanced frame geometry (adjacent)"


@dataclass(frozen=True)
class BalanceCriteria:
    """The limits and clause references a balance run is judged against."""

    k_ratio_min: float = 0.75
    T_ratio_min: float = 0.70
    mass_normalized: bool = True
    ref_stiffness: str = ""
    ref_geometry: str = ""

    @property
    def kappa_symbol(self) -> str:
        return "k/m" if self.mass_normalized else "k"


@dataclass
class BentStiffness:
    """One pier's balance-relevant state, evaluated at every fixity bound."""

    name: str
    frame: str
    order: int                       # position along the frame (table row order)
    Hcol: float                      # entered clear height, in
    silo: float                      # column silo depth, in
    mass: float                      # tributary seismic mass, kip*s^2/in
    k: tuple[float, ...]             # effective lateral stiffness per bound, kip/in
    T: tuple[float, ...]             # effective period per bound, s
    bound_labels: tuple[str, ...] = ()

    @property
    def H_free(self) -> float:
        return self.Hcol + self.silo

    def kappa(self, bound: int, mass_normalized: bool = True) -> float:
        """Stiffness (or stiffness-to-mass) ratio parameter at ``bound``."""
        k = self.k[bound]
        if not mass_normalized:
            return k
        return k / self.mass if self.mass > 0 else float("nan")

    def label(self, bound: int) -> str:
        if bound < len(self.bound_labels) and self.bound_labels[bound]:
            return self.bound_labels[bound]
        return f"bound {bound + 1}"


@dataclass
class BalanceCheck:
    """One code check on one adjacent pair at one fixity bound."""

    name: str
    pair: tuple[str, str]
    bound: str
    ratio: float
    limit: float
    passed: bool
    ref: str = ""
    note: str = ""

    @property
    def label(self) -> str:
        return f"{self.name}: {self.pair[0]}-{self.pair[1]} [{self.bound}]"


@dataclass
class BalanceResult:
    """Outcome of the balance stage for a whole batch."""

    bents: list[BentStiffness] = field(default_factory=list)
    checks: list[BalanceCheck] = field(default_factory=list)
    criteria: BalanceCriteria = field(default_factory=BalanceCriteria)
    log: list[str] = field(default_factory=list)
    converged: bool = True

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed(self) -> list[BalanceCheck]:
        return [c for c in self.checks if not c.passed]

    def worst_ratio(self, name: str, check_name: str) -> float | None:
        """Worst (lowest) ratio of ``check_name`` involving pier ``name``."""
        vals = [c.ratio for c in self.checks
                if c.name == check_name and name in c.pair
                and math.isfinite(c.ratio)]
        return min(vals) if vals else None

    def pier_passed(self, name: str) -> bool:
        """True if every check involving pier ``name`` passed."""
        return all(c.passed for c in self.checks if name in c.pair)


# ---------------------------------------------------------------------------
# Building bents from batch results
# ---------------------------------------------------------------------------
def bent_stiffness(name: str, frame: str, order: int, assessment,
                   Hcol: float, silo: float) -> BentStiffness:
    """Collect a row's stiffnesses/periods straight off its assessment bounds.

    No new mechanics: ``k`` is the effective (cracked, ``EI = Mp/phi_y``) lateral
    stiffness of the two-segment equivalent cantilever that already drives the
    displacement demand, and ``T`` the effective period that goes with it.
    """
    bounds = assessment.bounds
    return BentStiffness(
        name=name, frame=frame, order=order, Hcol=Hcol, silo=silo,
        mass=bounds[0].demand.mass if bounds else float("nan"),
        k=tuple(b.stiffness for b in bounds),
        T=tuple(b.demand.period for b in bounds),
        bound_labels=tuple(b.soil_label or f"{b.multiplier:.2g}·D_shaft"
                           for b in bounds),
    )


def adjacent_pairs(bents: list[BentStiffness]
                   ) -> list[tuple[BentStiffness, BentStiffness]]:
    """Consecutive pairs within each frame, in table row order.

    Bents are grouped by ``frame`` (order of first appearance preserved) and
    paired with their immediate neighbour.  A frame holding a single bent yields
    no pairs.  Callers exclude opted-out piers before calling.
    """
    groups: dict[str, list[BentStiffness]] = {}
    for b in sorted(bents, key=lambda b: b.order):
        groups.setdefault(b.frame, []).append(b)
    pairs = []
    for members in groups.values():
        pairs.extend(zip(members, members[1:]))
    return pairs


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------
def _ratio(a: float, b: float) -> float:
    """``min/max`` of two positive quantities; nan if either is unusable."""
    if not (math.isfinite(a) and math.isfinite(b)) or a <= 0.0 or b <= 0.0:
        return float("nan")
    return min(a, b) / max(a, b)


def balance_checks(bents: list[BentStiffness],
                   criteria: BalanceCriteria) -> list[BalanceCheck]:
    """Every adjacent-pair balance check, at every fixity bound.

    Bounds are compared **like for like** (stiff-vs-stiff, soft-vs-soft) so the
    same modelling assumption is applied to both piers of a pair, and every bound
    must comply.
    """
    checks: list[BalanceCheck] = []
    for bi, bj in adjacent_pairs(bents):
        n_bounds = min(len(bi.k), len(bj.k))
        for b in range(n_bounds):
            label = bi.label(b)
            ki = bi.kappa(b, criteria.mass_normalized)
            kj = bj.kappa(b, criteria.mass_normalized)
            r = _ratio(ki, kj)
            bad = math.isnan(r)
            checks.append(BalanceCheck(
                name=STIFFNESS_CHECK, pair=(bi.name, bj.name), bound=label,
                ratio=r, limit=criteria.k_ratio_min,
                passed=(not bad) and r >= criteria.k_ratio_min,
                ref=criteria.ref_stiffness,
                note=("stiffness unavailable (unstable p-y solve?)" if bad else
                      f"{criteria.kappa_symbol} = {ki:.4g} vs {kj:.4g}"),
            ))
            rT = _ratio(bi.T[b], bj.T[b])
            badT = math.isnan(rT)
            checks.append(BalanceCheck(
                name=GEOMETRY_CHECK, pair=(bi.name, bj.name), bound=label,
                ratio=rT, limit=criteria.T_ratio_min,
                passed=(not badT) and rT >= criteria.T_ratio_min,
                ref=criteria.ref_geometry,
                note=("period unavailable" if badT else
                      f"T = {bi.T[b]:.3f} s vs {bj.T[b]:.3f} s"),
            ))
    return checks


# ---------------------------------------------------------------------------
# Sizing a column silo
# ---------------------------------------------------------------------------
def stiffness_at_silo(geometry: Geometry, EI_col: float, EI_shaft: float,
                      multiplier: float, silo: float) -> float:
    """Elastic lateral stiffness of the two-segment cantilever at silo ``silo``.

    Reuses :meth:`Geometry.lateral_stiffness` with the silo swapped in, holding
    ``EI`` fixed.  Monotonically decreasing in ``silo``.
    """
    return replace(geometry, silo=silo).lateral_stiffness(
        EI_col, EI_shaft, multiplier)


def required_silo(geometry: Geometry, EI_col: float, EI_shaft: float,
                  multiplier: float, k_target: float,
                  silo_min: float = 0.0, silo_max: float = 240.0,
                  tol: float = 1e-3) -> float | None:
    """Silo depth (in) that softens this pier to ``k_target`` (kip/in).

    Bisection on the elastic two-segment cantilever — no moment-curvature and no
    p-y solve, so the orchestration can predict a depth cheaply and then verify
    it with one real re-analysis.  Returns ``None`` when ``k_target`` cannot be
    reached at or below ``silo_max`` (the cap binds), and ``silo_min`` when the
    pier is already soft enough.

    ``EI`` is held at its current value; growing the silo changes the demands and
    so may change the reinforcement (hence ``EI``), which is why the caller must
    re-run the seismic checks and iterate.
    """
    if silo_max <= silo_min:
        return silo_min if stiffness_at_silo(
            geometry, EI_col, EI_shaft, multiplier, silo_min) <= k_target else None
    if stiffness_at_silo(geometry, EI_col, EI_shaft, multiplier, silo_min) <= k_target:
        return silo_min
    if stiffness_at_silo(geometry, EI_col, EI_shaft, multiplier, silo_max) > k_target:
        return None                                   # unreachable within the cap
    lo, hi = silo_min, silo_max                       # k(lo) > target >= k(hi)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if stiffness_at_silo(geometry, EI_col, EI_shaft, multiplier, mid) > k_target:
            lo = mid
        else:
            hi = mid
    return hi


def mass_ratio_window(criteria: BalanceCriteria) -> tuple[float, float]:
    """Adjacent tributary-mass ratios for which BOTH rules can hold at once.

    Only meaningful when mass normalisation is OFF.  With ``kappa = k`` the two
    rules constrain the same quantity ``x = k_i/k_j`` from opposite directions::

        stiffness :  L_k <= x <= 1/L_k
        period    :  T_i/T_j = sqrt(mu/x)  with  mu = m_i/m_j
                     L_T <= sqrt(mu/x) <= 1/L_T   ->   mu*L_T^2 <= x <= mu/L_T^2

    Those two windows overlap only while ``L_k*L_T^2 <= mu <= 1/(L_k*L_T^2)`` —
    at the code limits, a factor of 2.72.  Outside it the pair cannot satisfy
    both clauses at ANY stiffness, so no silo (indeed no column change of any
    kind) will fix it: the tributary masses themselves have to move, or mass
    normalisation has to be switched on.

    With normalisation ON this can never bite — the kappa rule then implies the
    period rule — so the window is unbounded.
    """
    if criteria.mass_normalized:
        return (0.0, float("inf"))
    f = criteria.k_ratio_min * criteria.T_ratio_min ** 2
    return (f, 1.0 / f)


def joint_feasible(bi: BentStiffness, bj: BentStiffness,
                   criteria: BalanceCriteria) -> tuple[bool, float, tuple[float, float]]:
    """``(feasible, mass_ratio, allowed_k_ratio_window)`` for one adjacent pair."""
    lo, hi = mass_ratio_window(criteria)
    if bj.mass <= 0 or bi.mass <= 0:
        return True, float("nan"), (criteria.k_ratio_min, 1.0 / criteria.k_ratio_min)
    mu = bi.mass / bj.mass
    if criteria.mass_normalized:
        return True, mu, (criteria.k_ratio_min, 1.0 / criteria.k_ratio_min)
    t2 = criteria.T_ratio_min ** 2
    x_lo = max(criteria.k_ratio_min, mu * t2)
    x_hi = min(1.0 / criteria.k_ratio_min, mu / t2)
    return (lo <= mu <= hi), mu, (x_lo, x_hi)


def dedupe(lines: list[str]) -> list[str]:
    """Drop repeated lines, keeping first-seen order.

    The silo planner sweeps its pairs several times, so the same "cap reached"
    note can be raised many times over for one pair.
    """
    seen: set[str] = set()
    out = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out


def quantise_silo(silo: float, step: float, cap: float, floor: float = 0.0) -> float:
    """Round ``silo`` UP to a whole ``step``, clamped to ``[floor, cap]``.

    ``floor`` wins over ``cap``: the floor is what the user entered, and the cap
    only limits what the tool may *add*.  Never returns less than ``floor``.
    """
    if step > 0.0:
        silo = math.ceil(silo / step - 1e-9) * step
    return min(max(silo, floor), max(cap, floor))
