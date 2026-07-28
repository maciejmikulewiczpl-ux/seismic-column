"""Balanced stiffness and balanced frame geometry between adjacent frames.

A bridge is a run of **frames**, and the two code rules act at different levels:

*Balanced stiffness* — Caltrans SDC 2.1 §7.1.2 Table 7.1.2-1, AASHTO SGS 3rd Ed.
§4.1.2 — compares bents **inside one frame**::

    adjacent bents in a frame   min(kappa_i, kappa_j)/max(...) >= 0.75
    any two bents in a frame    min(kappa_i, kappa_j)/max(...) >= 0.50

with ``kappa = k_e/m`` (the stiffness-to-mass form Caltrans always uses, and the
AASHTO variable-width form) or ``kappa = k_e`` (AASHTO constant width).  A frame
holding a single bent has nothing to compare, so **no stiffness rule applies to a
run of simply supported spans** — each of those is its own frame.

*Balanced frame geometry* — SDC §7.1.3, SGS §4.1.3 — compares **adjacent
frames**::

    min(T_i, T_j) / max(T_i, T_j) >= 0.70

and applies everywhere, simply supported or continuous.

Writing both as ``min/max`` makes Caltrans' two-sided limits (0.75…1.33,
0.5…2.0, 0.7…1.43) and AASHTO's one-sided ones the same test, and makes each
check independent of which member is called *i*.

Direction matters
-----------------
Everything above is evaluated **longitudinally and transversely**, because the
tributary mass a bent restrains differs by direction and so, therefore, do the
periods.  Which bents form a frame also differs by direction: a bearing that is
released longitudinally but shear-keyed transversely joins the frame one way and
not the other.  :func:`frames_for` derives both layouts from ``deck_link``.

A frame's period is the frame's, not a bent's::

    K_frame = sum of k over the members that resist in this direction
    M_frame = sum of their tributary mass
    T_frame = 2*pi*sqrt(M_frame / K_frame)

which is the standard rigid-deck, stand-alone-frame ESA idealisation.  A
single-bent frame reduces to exactly the bent's own SDOF period.

Member stiffness is the **fixed-free** two-segment cantilever already computed
for the seismic run — correct transversely, and for every simply supported bent.
An integral bent is a fixed moment connection and is therefore fixed-fixed
*longitudinally* (SDC C7.1.2 gives 12EcIeff/L^3 against 3EcIeff/L^3), which this
module does not yet model; see :data:`END_CONDITION_NOTE`.

The tuning lever is the **column silo** (isolation casing), which lengthens the
free column and so softens a stiff pier — Caltrans C7.1.2 / SGS §4.1.4.
:func:`required_silo` sizes it from the elastic two-segment cantilever, which
costs nothing (no moment-curvature, no p-y), so the batch orchestration can
predict a depth before paying for a full re-analysis at that depth.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace

from .geometry import Geometry
from .io_schema import DIRECTIONS, LONGITUDINAL, TRANSVERSE, in_frame

STIFFNESS_CHECK = "Balanced stiffness (adjacent)"
STIFFNESS_ANY_CHECK = "Balanced stiffness (any two)"
GEOMETRY_CHECK = "Balanced frame geometry (adjacent)"

# Stated wherever a frame period is reported, because it is an approximation
# that is NOT conservative in a known direction.
END_CONDITION_NOTE = (
    "Member stiffness is the fixed-free cantilever (SDC C7.1.2-1, 3EcIeff/L³). "
    "That is right transversely and for every simply supported bent, but an "
    "integral bent is a fixed moment connection and so is fixed-fixed "
    "longitudinally (C7.1.2-2, 12EcIeff/L³). Longitudinal K_frame is therefore "
    "understated and T_long overstated for a continuous frame."
)


@dataclass(frozen=True)
class BalanceCriteria:
    """The limits and clause references a balance run is judged against."""

    k_ratio_min: float = 0.75          # adjacent bents within a frame
    k_ratio_any: float = 0.50          # any two bents within a frame
    T_ratio_min: float = 0.70          # adjacent frames
    mass_normalized: bool = True
    ref_stiffness: str = ""
    ref_stiffness_any: str = ""
    ref_geometry: str = ""

    @property
    def kappa_symbol(self) -> str:
        return "k/m" if self.mass_normalized else "k"


@dataclass
class BentStiffness:
    """One pier's balance-relevant state, evaluated at every fixity bound.

    ``k`` is direction-independent — a circular column on a circular shaft is
    axisymmetric — so only the mass, and hence the period and ``kappa``, differ
    between longitudinal and transverse.
    """

    name: str
    frame: str
    order: int                       # position along the bridge (table row order)
    Hcol: float                      # entered clear height, in
    silo: float                      # column silo depth, in
    k: tuple[float, ...]             # effective lateral stiffness per bound, kip/in
    mass_long: float = float("nan")  # tributary mass restrained longitudinally
    mass_trans: float = float("nan")  # ... and transversely, kip*s^2/in
    deck_link: str = "integral"
    bound_labels: tuple[str, ...] = ()

    @property
    def H_free(self) -> float:
        return self.Hcol + self.silo

    def mass(self, direction: str) -> float:
        """Tributary mass restrained in ``direction``, kip*s^2/in."""
        return self.mass_long if direction == LONGITUDINAL else self.mass_trans

    def participates(self, direction: str) -> bool:
        """Does this bent resist the deck in ``direction``?

        ``integral`` is monolithic and resists both ways; ``bearing`` is released
        longitudinally but shear-keyed, so it resists transversely only; ``free``
        resists neither.
        """
        if self.deck_link == "free":
            return False
        if self.deck_link == "bearing":
            return direction == TRANSVERSE
        return True                                   # integral

    def T(self, direction: str, bound: int) -> float:
        """Effective SDOF period of this bent alone in ``direction``, s."""
        k, m = self.k[bound], self.mass(direction)
        if not (math.isfinite(k) and math.isfinite(m)) or k <= 0 or m <= 0:
            return float("nan")
        return 2.0 * math.pi * math.sqrt(m / k)

    def kappa(self, direction: str, bound: int,
              mass_normalized: bool = True) -> float:
        """Stiffness (or stiffness-to-mass) ratio parameter at ``bound``."""
        k = self.k[bound]
        if not mass_normalized:
            return k
        m = self.mass(direction)
        return k / m if m > 0 else float("nan")

    def label(self, bound: int) -> str:
        if bound < len(self.bound_labels) and self.bound_labels[bound]:
            return self.bound_labels[bound]
        return f"bound {bound + 1}"


@dataclass
class Frame:
    """The bents that act together in one direction, and their frame period."""

    key: str
    direction: str
    members: list[BentStiffness]
    order: int                       # position along the bridge (lowest member)

    @property
    def continuous(self) -> bool:
        """More than one bent acts together, so the stiffness rules apply."""
        return len(self.members) > 1

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(b.name for b in self.members)

    @property
    def n_bounds(self) -> int:
        return min((len(b.k) for b in self.members), default=0)

    def K(self, bound: int) -> float:
        """Frame lateral stiffness, kip/in — rigid deck, so the members add."""
        return sum(b.k[bound] for b in self.members)

    def M(self) -> float:
        """Frame tributary mass in this direction, kip*s^2/in."""
        return sum(b.mass(self.direction) for b in self.members)

    def T(self, bound: int) -> float:
        """Frame period, s.  Reduces to the bent's own period when alone."""
        K, M = self.K(bound), self.M()
        if not (math.isfinite(K) and math.isfinite(M)) or K <= 0 or M <= 0:
            return float("nan")
        return 2.0 * math.pi * math.sqrt(M / K)

    def label(self, bound: int) -> str:
        return self.members[0].label(bound) if self.members else f"bound {bound+1}"


@dataclass
class BalanceCheck:
    """One code check on one pair, in one direction, at one fixity bound."""

    name: str
    pair: tuple[str, str]
    bound: str
    ratio: float
    limit: float
    passed: bool
    direction: str = ""
    scope: str = ""                  # frame key the check belongs to
    ref: str = ""
    note: str = ""

    @property
    def label(self) -> str:
        d = f", {self.direction[:5]}." if self.direction else ""
        return f"{self.name}: {self.pair[0]}-{self.pair[1]} [{self.bound}{d}]"


@dataclass
class BalanceResult:
    """Outcome of the balance stage for a whole batch."""

    bents: list[BentStiffness] = field(default_factory=list)
    checks: list[BalanceCheck] = field(default_factory=list)
    criteria: BalanceCriteria = field(default_factory=BalanceCriteria)
    frames: dict[str, list[Frame]] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)
    converged: bool = True

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed(self) -> list[BalanceCheck]:
        return [c for c in self.checks if not c.passed]

    def worst_ratio(self, name: str, check_name: str,
                    direction: str | None = None) -> float | None:
        """Worst (lowest) ratio of ``check_name`` involving ``name``.

        ``name`` matches either a pier or, for the geometry rule, a frame key.
        """
        vals = [c.ratio for c in self.checks
                if c.name == check_name and name in c.pair
                and (direction is None or c.direction == direction)
                and math.isfinite(c.ratio)]
        return min(vals) if vals else None

    def frames_of(self, pier: str) -> set[str]:
        """Frame keys the pier belongs to, across both directions."""
        return {f.key for frames in self.frames.values() for f in frames
                if pier in f.names}

    def pier_passed(self, name: str) -> bool:
        """True if every check touching this pier — or its frame — passed."""
        keys = self.frames_of(name) | {name}
        return all(c.passed for c in self.checks if keys & set(c.pair))

    def touches(self, name: str) -> bool:
        """Is this pier involved in any check at all?"""
        keys = self.frames_of(name) | {name}
        return any(keys & set(c.pair) for c in self.checks)


# ---------------------------------------------------------------------------
# Building bents and frames from batch results
# ---------------------------------------------------------------------------
def bent_stiffness(name: str, frame: str, order: int, assessment,
                   Hcol: float, silo: float, mass_long: float,
                   mass_trans: float, deck_link: str = "integral"
                   ) -> BentStiffness:
    """Collect a row's stiffnesses straight off its assessment bounds.

    No new mechanics: ``k`` is the effective (cracked, ``EI = Mp/phi_y``) lateral
    stiffness of the two-segment equivalent cantilever that already drives the
    displacement demand.  Periods are derived per direction from the two masses.
    """
    bounds = assessment.bounds
    labels = [b.soil_label or f"{b.multiplier:.2g}·D_shaft" for b in bounds]
    # Collapse duplicate bounds.  Setting both soil brackets to the same factor
    # (or both fixity multipliers to the same value) makes evaluate_column run
    # the identical analysis twice, which would otherwise double every check,
    # every table row and every legend entry for no information.
    keep, seen = [], set()
    for i, lbl in enumerate(labels):
        if lbl in seen:
            continue
        seen.add(lbl)
        keep.append(i)
    return BentStiffness(
        name=name, frame=frame, order=order, Hcol=Hcol, silo=silo,
        k=tuple(bounds[i].stiffness for i in keep),
        mass_long=mass_long, mass_trans=mass_trans, deck_link=deck_link,
        bound_labels=tuple(labels[i] for i in keep),
    )


def frames_for(bents: list[BentStiffness], direction: str) -> list[Frame]:
    """The frames acting in ``direction``, ordered along the bridge.

    Within a ``frame`` id, the bents that resist in this direction act together.
    A bent that does **not** resist here still holds whatever tributary mass it
    does restrain, so it splits off as its own single-bent frame — which is how a
    longitudinally-released bearing behaves: out of the continuous frame one way,
    inside it the other::

        LONGITUDINAL  A6 | A7 | [A8 A9 A10] | A11 | A12      (A7/A11 released)
        TRANSVERSE    A6 | [A7 A8 A9 A10 A11] | A12          (shear keys engage)

    A bent with no mass in this direction drops out entirely, as does one opted
    out of the checks via a blank ``frame``.
    """
    groups: dict[str, list[BentStiffness]] = {}
    for b in sorted(bents, key=lambda b: b.order):
        if not in_frame(b.frame):
            continue
        groups.setdefault(b.frame, []).append(b)

    frames: list[Frame] = []
    for key, members in groups.items():
        holds = [b for b in members if b.mass(direction) > 0]
        acting = [b for b in holds if b.participates(direction)]
        # A bearing released in THIS direction still holds whatever span it is
        # fixed to, so it stands alone.  A `free` bent resists nothing at all
        # and leaves the model entirely.
        standalone = [b for b in holds
                      if not b.participates(direction) and b.deck_link != "free"]
        if acting:
            frames.append(Frame(key=key, direction=direction, members=acting,
                                order=min(b.order for b in acting)))
        for b in standalone:
            frames.append(Frame(key=f"{key}·{b.name}", direction=direction,
                                members=[b], order=b.order))
    frames.sort(key=lambda f: f.order)
    return frames


def adjacent_pairs(bents: list[BentStiffness]
                   ) -> list[tuple[BentStiffness, BentStiffness]]:
    """Consecutive bents within each frame id, in table row order.

    Retained for the silo planner and the mass-window diagnostic, which reason
    about bents rather than frames.  A frame holding a single bent yields no
    pairs.
    """
    groups: dict[str, list[BentStiffness]] = {}
    for b in sorted(bents, key=lambda b: b.order):
        if not in_frame(b.frame):
            continue
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
    """Every balance check, in both directions, at every fixity bound.

    Bounds are compared **like for like** (stiff-vs-stiff, soft-vs-soft) so the
    same modelling assumption is applied to both sides of a pair, and every bound
    must comply.

    The any-two rule is reported for **non-adjacent** pairs only: an adjacent
    pair is also "any two", but its 0.75 limit is stricter than 0.50, so the
    looser check can never govern there and printing it twice is noise.
    """
    checks: list[BalanceCheck] = []
    for direction in DIRECTIONS:
        frames = frames_for(bents, direction)

        # --- stiffness: inside a continuous frame only ---
        for f in frames:
            if not f.continuous:
                continue
            adjacent = {(a.name, b.name) for a, b in zip(f.members, f.members[1:])}
            for bound in range(f.n_bounds):
                label = f.label(bound)
                for bi, bj in itertools.combinations(f.members, 2):
                    near = (bi.name, bj.name) in adjacent
                    ki = bi.kappa(direction, bound, criteria.mass_normalized)
                    kj = bj.kappa(direction, bound, criteria.mass_normalized)
                    r = _ratio(ki, kj)
                    bad = math.isnan(r)
                    limit = criteria.k_ratio_min if near else criteria.k_ratio_any
                    checks.append(BalanceCheck(
                        name=STIFFNESS_CHECK if near else STIFFNESS_ANY_CHECK,
                        pair=(bi.name, bj.name), bound=label,
                        ratio=r, limit=limit,
                        passed=(not bad) and r >= limit,
                        direction=direction, scope=f.key,
                        ref=(criteria.ref_stiffness if near
                             else criteria.ref_stiffness_any),
                        note=("stiffness unavailable (unstable p-y solve?)" if bad
                              else f"{criteria.kappa_symbol} = {ki:.4g} vs {kj:.4g}"),
                    ))

        # --- geometry: between adjacent frames, everywhere ---
        for fi, fj in zip(frames, frames[1:]):
            for bound in range(min(fi.n_bounds, fj.n_bounds)):
                Ti, Tj = fi.T(bound), fj.T(bound)
                r = _ratio(Ti, Tj)
                bad = math.isnan(r)
                checks.append(BalanceCheck(
                    name=GEOMETRY_CHECK, pair=(fi.key, fj.key),
                    bound=fi.label(bound), ratio=r, limit=criteria.T_ratio_min,
                    passed=(not bad) and r >= criteria.T_ratio_min,
                    direction=direction, scope="",
                    ref=criteria.ref_geometry,
                    note=("period unavailable" if bad else
                          f"T = {Ti:.3f} s vs {Tj:.3f} s"),
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


# NOTE: a "joint feasibility" diagnostic used to live here.  It solved the case
# where the stiffness rule and the period rule bound the SAME k ratio from
# opposite sides, which could make a pair unsatisfiable at any stiffness.  That
# clash cannot arise any more: the stiffness rule acts on bents INSIDE a frame
# and the period rule on adjacent FRAMES, so the two never constrain the same
# pair.  It was removed rather than left as a check that can never fire.


# ---------------------------------------------------------------------------
# Exact minimum-silo assignment
# ---------------------------------------------------------------------------
@dataclass
class SiloPlan:
    """Result of a silo search."""

    silos: dict[str, float]                  # pier name -> silo depth, in
    feasible: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def total(self) -> float:
        return sum(self.silos.values())

    @property
    def deepest(self) -> float:
        return max(self.silos.values(), default=0.0)


def silo_states(floor: float, cap: float, step: float) -> tuple[float, ...]:
    """Buildable silo depths, in: ``floor`` then whole ``step``s up to ``cap``.

    The floor (an entered ``silo_ft``) is always allowed even when it is not on
    the grid — it is the engineer's number and must not be rounded away.
    """
    if step <= 0.0:
        return (floor,)
    out = [floor]
    first = math.ceil(floor / step + 1e-9) * step
    z = first
    while z <= cap + 1e-9:
        if z > floor + 1e-9:
            out.append(z)
        z += step
    return tuple(out)


def pair_ok(ki: float, kj: float, mi: float, mj: float,
            criteria: BalanceCriteria, k_limit: float | None = None) -> bool:
    """Do two piers satisfy BOTH balance rules at this bound?

    ``k_limit`` defaults to the adjacent-bent limit; pass ``k_ratio_any`` for a
    non-adjacent pair inside a frame.  The period limit is the same either way.
    """
    if not (math.isfinite(ki) and math.isfinite(kj)) or ki <= 0 or kj <= 0:
        return False
    ai = ki / mi if criteria.mass_normalized else ki
    aj = kj / mj if criteria.mass_normalized else kj
    if _ratio(ai, aj) < (criteria.k_ratio_min if k_limit is None else k_limit):
        return False
    Ti = math.sqrt(mi / ki)                 # 2*pi cancels in the ratio
    Tj = math.sqrt(mj / kj)
    return _ratio(Ti, Tj) >= criteria.T_ratio_min


def dp_min_silo(bents: list[BentStiffness], states: dict[str, tuple[float, ...]],
                feasible, chain: list[list[BentStiffness]] | None = None
                ) -> SiloPlan:
    """Minimum-total-silo assignment on the buildable grid — exact, per chain.

    Silo depths are discrete (whole ``silo_step_ft`` increments up to the cap).
    When the governing rules only couple *neighbours*, each run of bents is a
    chain and the cheapest feasible assignment over a chain is a textbook
    dynamic program::

        dp[i][s] = s + min over s' allowed by the pair (i-1, i) of dp[i-1][s']

    That is the true optimum on the grid, not a relaxation of it — unlike the
    pairwise repair, which fixes one failing pair at a time and pays for the
    cascade each local fix sets off in its neighbour.

    ``feasible(bi, si, bj, sj) -> bool`` decides whether neighbouring bents may
    sit at those two depths; the caller owns which rules that covers and is
    responsible for calibrating its stiffness predictions against a real
    analysis and iterating.  ``chain`` gives the runs to solve (default: group by
    ``frame``); a caller whose rules couple frames rather than bents passes the
    bridge-order run instead.

    Cost is O(n · S²).  Returns ``feasible=False`` with the offending pair named
    when some neighbouring pair admits no combination at all.

    **The caller must not use this when the rules are not neighbour-only** — the
    any-two-bents rule makes a continuous frame all-pairs, which this cannot
    represent.
    """
    silos: dict[str, float] = {}
    notes: list[str] = []
    ok = True

    if chain is None:
        groups: dict[str, list[BentStiffness]] = {}
        for b in sorted(bents, key=lambda b: b.order):
            groups.setdefault(b.frame, []).append(b)
        chain = list(groups.values())

    for members in chain:
        if len(members) == 1:                 # nothing to balance against
            silos[members[0].name] = states[members[0].name][0]
            continue
        n = len(members)
        st = [states[b.name] for b in members]
        # dp[s] = (cost, backpointer) for the pier being processed
        dp = [(s, -1) for s in st[0]]
        back: list[list[int]] = []
        for i in range(1, n):
            bi, bj = members[i - 1], members[i]
            row, bp = [], []
            for sj in st[i]:
                best, arg = math.inf, -1
                for a, si in enumerate(st[i - 1]):
                    if dp[a][0] == math.inf:
                        continue
                    if feasible(bi, si, bj, sj):
                        if dp[a][0] < best:
                            best, arg = dp[a][0], a
                row.append((best + sj if best < math.inf else math.inf, arg))
                bp.append(arg)
            back.append(bp)
            dp = row
            if all(c == math.inf for c, _ in dp):
                ok = False
                deepest = max(st[i - 1][-1], st[i][-1]) / 12.0
                notes.append(
                    f"INFEASIBLE — {bi.name}-{bj.name}: the {deepest:g} ft silo "
                    f"cap is reached and no combination of buildable depths "
                    f"satisfies both balance rules for this pair. A silo only "
                    f"softens, so stiffen the more flexible pier (larger "
                    f"column), rebalance their tributary masses, or raise the "
                    f"cap.")
                break
        else:
            end = min(range(len(dp)), key=lambda a: dp[a][0])
            if dp[end][0] == math.inf:
                ok = False
            else:
                idx = [0] * n
                idx[n - 1] = end
                for i in range(n - 1, 0, -1):
                    idx[i - 1] = back[i - 1][idx[i]]
                for i, b in enumerate(members):
                    silos[b.name] = st[i][idx[i]]
                continue
        for i, b in enumerate(members):       # infeasible frame: leave at floor
            silos.setdefault(b.name, st[i][0])

    return SiloPlan(silos=silos, feasible=ok, notes=notes)


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
