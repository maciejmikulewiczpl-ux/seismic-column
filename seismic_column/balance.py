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

Member stiffness follows the **end condition**, per direction.  An integral
bent is a moment connection, so longitudinally it is fixed-fixed — SDC C7.1.2-2
``12EcIeff/L^3`` against C7.1.2-1 ``3EcIeff/L^3``, a factor of 4 on a prismatic
member.  Transversely even an integral single-column bent behaves as a
cantilever, and a bent on bearings is fixed-free either way.  See
:meth:`Geometry.tip_flexibility` for the stepped two-segment solution.

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
from .io_schema import (DIRECTIONS, LONGITUDINAL, TRANSVERSE, deck_links,
                        frame_keys,
                        head_moment_connection, in_frame)

STIFFNESS_CHECK = "Balanced stiffness (adjacent)"
STIFFNESS_ANY_CHECK = "Balanced stiffness (any two)"
GEOMETRY_CHECK = "Balanced frame geometry (adjacent)"

# Stated wherever a frame period is reported, because it is an approximation
# that is NOT conservative in a known direction.
END_CONDITION_NOTE = (
    "Member stiffness follows the end condition: an integral (moment-connected) "
    "bent is FIXED-FIXED longitudinally (SDC C7.1.2-2, 12EcIeff/L³ prismatic), "
    "while a bent on bearings, and any bent transversely, is fixed-free "
    "(C7.1.2-1, 3EcIeff/L³). The depth to fixity Df is itself derived for a "
    "free-head member, so re-using it with a fixed head is the usual "
    "simplification rather than an exact equivalence."
)


@dataclass(frozen=True)
class BalanceCriteria:
    """The limits and clause references a balance run is judged against."""

    k_ratio_min: float = 0.75          # adjacent bents within a frame
    k_ratio_any: float = 0.50          # any two bents within a frame
    T_ratio_min: float = 0.70          # adjacent frames
    mass_normalized: bool = True
    # Whether the two piers carrying one SIMPLY SUPPORTED span are checked
    # against each other for balanced stiffness.  Off by default: a run of
    # simple spans is modelled one frame per pier, and a frame holding a single
    # pier has nothing to compare -- they are matched on period alone.  Turn it
    # on to read each span as a frame whose two supports must also be balanced.
    simple_span_stiffness: bool = False
    ref_stiffness: str = ""
    ref_stiffness_any: str = ""
    ref_geometry: str = ""

    @property
    def kappa_symbol(self) -> str:
        return "k/m" if self.mass_normalized else "k"


@dataclass
class BentStiffness:
    """One pier's balance-relevant state, evaluated at every fixity bound.

    The SECTION is axisymmetric — a circular column on a circular shaft — so
    the stiffness differs by direction only through the **end condition**: an
    integral (moment-connected) bent is fixed-fixed longitudinally, everything
    else is a fixed-free cantilever.  Mass differs by direction too, so the
    period and ``kappa`` do on both counts.
    """

    name: str
    frame: str
    order: int                       # position along the bridge (table row order)
    Hcol: float                      # entered clear height, in
    silo: float                      # column silo depth, in
    k: tuple[float, ...]             # FIXED-FREE stiffness per bound, kip/in
    mass_long: float = float("nan")  # tributary mass restrained longitudinally
    mass_trans: float = float("nan")  # ... and transversely, kip*s^2/in
    deck_link: str = "pinned"
    n_columns: int = 1            # columns in this bent; they act in parallel
    cap_fixity: str = "fixed"     # transverse column-to-cap: fixed | pinned
    bound_labels: tuple[str, ...] = ()
    # Fixed-FIXED stiffness per bound, used longitudinally for an integral bent.
    # Defaults to the fixed-free values when the caller has not computed it.
    k_fixed: tuple[float, ...] = ()

    @property
    def H_free(self) -> float:
        return self.Hcol + self.silo

    def mass(self, direction: str) -> float:
        """Tributary mass restrained in ``direction``, kip*s^2/in."""
        return self.mass_long if direction == LONGITUDINAL else self.mass_trans

    @property
    def frames(self) -> tuple[str, ...]:
        """Every frame this bent belongs to -- more than one when it sits under
        an expansion joint and carries a deck either side."""
        return frame_keys(self.frame)

    @property
    def links(self) -> tuple[str, ...]:
        """How this bent meets each frame it carries, aligned with :attr:`frames`."""
        return deck_links(self.deck_link, len(self.frames) or 1)

    def participates(self, direction: str, link: str | None = None) -> bool:
        """Does this bent resist a deck in ``direction`` through ``link``?

        ``integral`` is monolithic and resists both ways; ``bearing`` is released
        longitudinally but shear-keyed, so it resists transversely only; ``free``
        resists neither.  With no ``link`` given, ANY of its links counts -- the
        bent resists something in this direction.
        """
        if link is None:
            return any(self.participates(direction, ln) for ln in self.links)
        if link == "free":
            return False
        if link == "bearing":
            return direction == TRANSVERSE
        return True                                   # integral / pinned

    def end_fixity(self, direction: str) -> str:
        """Rotational restraint at the column head: ``'fixed'`` or ``'free'``.

        A SINGLE-column bent has no bent cap: it is either integral with the
        deck or sits on a bearing, which is exactly what ``deck_link`` records,
        so ``cap_fixity`` does not apply.  Longitudinally an integral bent is
        fixed; transversely it is a cantilever, having nothing to frame against.

        A MULTI-column bent meets a bent cap, and that connection can be a
        moment connection, a pin, or a pin in one direction only.  There the
        head is fixed only if BOTH halves are present:

        1. the COLUMN-TO-CAP connection transmits moment in this direction, and
        2. something above can resist it — longitudinally the deck has to be
           INTEGRAL (a cap beam is weak out of its own plane, so the cap alone
           cannot hold the column longitudinally); transversely the cap spans to
           the other columns, which is enough on its own.

        A pin releases the head whatever sits above it: an integral deck cannot
        hold a rotation the column-to-cap connection does not carry.
        """
        if self.n_columns <= 1:
            return ("fixed" if "integral" in self.links
                    and direction == LONGITUDINAL else "free")
        if not head_moment_connection(self.cap_fixity, direction):
            return "free"
        if direction == TRANSVERSE:
            return "fixed"
        return "fixed" if "integral" in self.links else "free"

    def stiffness(self, direction: str, bound: int) -> float:
        """Effective lateral stiffness of the BENT in ``direction``, kip/in.

        The columns of a bent act in parallel, so the bent is ``n`` times one
        column.  Per-column differences from the push/pull (each position has a
        slightly different Df) are second order here and are ignored: the
        balance rules compare bents with each other, not columns within a bent.
        """
        if self.end_fixity(direction) == "fixed" and bound < len(self.k_fixed):
            return self.n_columns * self.k_fixed[bound]
        return self.n_columns * self.k[bound]

    def T(self, direction: str, bound: int) -> float:
        """Effective SDOF period of this bent alone in ``direction``, s."""
        k, m = self.stiffness(direction, bound), self.mass(direction)
        if not (math.isfinite(k) and math.isfinite(m)) or k <= 0 or m <= 0:
            return float("nan")
        return 2.0 * math.pi * math.sqrt(m / k)

    def kappa(self, direction: str, bound: int,
              mass_normalized: bool = True) -> float:
        """Stiffness (or stiffness-to-mass) ratio parameter at ``bound``."""
        k = self.stiffness(direction, bound)
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
    # Fraction of each member's tributary mass credited to THIS frame.  A bent
    # under an expansion joint carries a half-span from either side, so it
    # appears in BOTH frames at 0.5.  Its STIFFNESS is full in both: each frame
    # is analysed on its own and a rigid deck leans on the whole bent.
    mass_share: dict[str, float] = field(default_factory=dict)
    # How many bents NAMED this frame, before any direction filtering.  A frame
    # declared by one bent is a simply supported span; one declared by several
    # is a continuous frame even if only one of them resists in this direction.
    declared: int = 0

    @property
    def simple_span(self) -> bool:
        return self.declared == 1

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
        """Frame lateral stiffness, kip/in — rigid deck, so the members add.

        Each member contributes at ITS end condition in this direction, so an
        integral bent brings its fixed-fixed stiffness longitudinally.
        """
        return sum(b.stiffness(self.direction, bound) for b in self.members)

    @property
    def end_conditions(self) -> str:
        """e.g. ``'fixed-fixed'`` or ``'fixed-free'``, or a mix."""
        kinds = {("fixed-fixed" if b.end_fixity(self.direction) == "fixed"
                  else "fixed-free") for b in self.members}
        return " / ".join(sorted(kinds))

    def M(self) -> float:
        """Frame tributary mass in this direction, kip*s^2/in.

        A boundary bent enters at its ``mass_share`` — the part of its tributary
        that belongs to THIS frame's deck.
        """
        return sum(b.mass(self.direction) * self.mass_share.get(b.name, 1.0)
                   for b in self.members)

    def shared(self, name: str) -> bool:
        """True when ``name`` is a boundary bent carrying two decks."""
        return self.mass_share.get(name, 1.0) < 1.0

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
    # Set only on a BETWEEN-FRAME geometry check that the silo search pursued
    # and could not close by practical means.  The balance rules exist to let a
    # bridge AVOID a more rigorous analysis (SDC 7.1.3 / SGS 4.1.3); where they
    # cannot be met, the code's own route is nonlinear time-history, not an
    # unbuildable silo.  A WITHIN-frame stiffness shortfall never gets this --
    # that rule governs how the frame itself behaves and has to be satisfied.
    tha_required: bool = False

    @property
    def status(self) -> str:
        if self.passed:
            return "OK"
        return "THA" if self.tha_required else "NG"

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
        """No HARD failure left.

        A geometry shortfall the search pursued and could not close by
        practical means is not counted here — it is referred to time-history
        analysis instead, which is the code's own route.  It is still reported,
        prominently, via :attr:`needs_tha`.
        """
        return all(c.passed or c.tha_required for c in self.checks)

    @property
    def failed(self) -> list[BalanceCheck]:
        """Hard failures only — the ones that must be designed out."""
        return [c for c in self.checks if not (c.passed or c.tha_required)]

    @property
    def needs_tha(self) -> list[BalanceCheck]:
        """Geometry pairs referred to time-history analysis."""
        return [c for c in self.checks if c.tha_required]

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
                   mass_trans: float, deck_link: str = "pinned",
                   D_shaft: float = 0.0, n_columns: int = 1,
                   cap_fixity: str = "fixed") -> BentStiffness:
    """Collect a row's stiffnesses straight off its assessment bounds.

    No new mechanics: ``k`` is the effective (cracked, ``EI = Mp/phi_y``) lateral
    stiffness of the two-segment equivalent cantilever that already drives the
    displacement demand.  Periods are derived per direction from the two masses.

    The two stiffnesses come from DIFFERENT bound sets, because the depth to
    fixity is not the same under the two head conditions: ``k`` (fixed-free,
    used transversely and by every non-integral bent) reads the transverse
    bounds, while ``k_fixed`` reads the longitudinal ones, where an integral
    bent's Df was solved with the head restrained at the mechanism shear.
    Taking both off one bound set would apply a fixed-head Df to the free-head
    stiffness, which is the error this whole distinction exists to avoid.
    """
    dirs = getattr(assessment, "directions", {}) or {}
    free_bounds = (dirs[TRANSVERSE].bounds if TRANSVERSE in dirs
                   else assessment.bounds)
    fixed_bounds = (dirs[LONGITUDINAL].bounds if LONGITUDINAL in dirs
                    else assessment.bounds)
    bounds = free_bounds
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
    # An integral bent is fixed-fixed longitudinally, so it needs the second
    # stiffness.  Same geometry, same cracked EI — only the head restraint
    # differs, so it costs one closed-form evaluation per bound.
    # A fixed head arises two ways: an integral bent longitudinally, or ANY
    # multi-column bent transversely (the cap restrains the heads).  Either way
    # the second stiffness is needed.
    k_fixed: tuple[float, ...] = ()
    # cap_fixity only applies to a bent that actually has a cap
    _multi = n_columns > 1
    _needs_fixed = (
        ("integral" in deck_links(deck_link, len(frame_keys(frame)) or 1)
         and (not _multi or head_moment_connection(cap_fixity, LONGITUDINAL)))
        or (_multi and head_moment_connection(cap_fixity, TRANSVERSE)))
    if _needs_fixed and D_shaft > 0:
        geom = Geometry(Hcol=Hcol, D_shaft=D_shaft, silo=silo)
        k_fixed = tuple(
            geom.lateral_stiffness(assessment.EI_col, assessment.EI_shaft,
                                   fixed_bounds[min(i, len(fixed_bounds) - 1)]
                                   .multiplier, end_fixity="fixed")
            for i in keep)
    return BentStiffness(
        name=name, frame=frame, order=order, Hcol=Hcol, silo=silo,
        k=tuple(bounds[i].stiffness for i in keep),
        mass_long=mass_long, mass_trans=mass_trans, deck_link=deck_link,
        bound_labels=tuple(labels[i] for i in keep), k_fixed=k_fixed,
        n_columns=n_columns, cap_fixity=cap_fixity,
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

    **A released bearing leaves the model in that direction.** It carries no
    deck there — the superstructure slides on it — so all it holds is its own
    cap and column self weight. It is not a frame, and pairing that self weight
    against a real frame's period is meaningless, so it takes no part in the
    balance checks longitudinally. Its own seismic checks are unaffected and
    still run on its own stand-alone period.

    **A boundary bent belongs to BOTH frames it carries.** An expansion joint
    over a bent means the deck each side bears on the same cap, so the bent
    supports the last span of one frame and the first span of the next. It
    therefore appears in both, bringing its FULL stiffness to each (each frame
    is analysed alone, and a rigid deck leans on the whole bent) and HALF its
    tributary mass (the half-span belonging to that frame's deck).
    """
    # Each (frame, link) pair the bent declares, kept together: the SAME bent
    # can meet two decks differently -- a free bearing under the end of a
    # continuous frame and a pin under the simple span beside it.
    groups: dict[str, list[BentStiffness]] = {}
    resisting: dict[str, int] = {}
    for b in sorted(bents, key=lambda b: b.order):
        keys, links = b.frames, b.links
        if not keys:
            continue
        # How many of its decks this bent actually resists in THIS direction.
        # Its tributary divides between exactly those: a pier that pins one
        # span longitudinally carries all of that span and none of the deck it
        # only bears, while transversely the shear keys engage both and it
        # takes half of each.
        n = sum(1 for ln in links if b.participates(direction, ln))
        resisting[b.name] = n
        for key in keys:
            groups.setdefault(key, []).append(b)

    frames: list[Frame] = []
    for key in sorted(groups, key=lambda k: min(b.order for b in groups[k])):
        acting = [b for b in groups[key]
                  if b.mass(direction) > 0 and resisting.get(b.name)
                  and b.participates(direction, dict(zip(b.frames,
                                                         b.links))[key])]
        if acting:
            frames.append(Frame(key=key, direction=direction,
                                members=list(acting),
                                order=min(b.order for b in acting),
                                declared=len(groups[key]),
                                mass_share={b.name: 1.0 / resisting[b.name]
                                            for b in acting}))

    # An expansion-joint bent sits at the END of one frame's deck and the START
    # of the next.  Give it to the neighbour too, at half its tributary mass.
    #
    # Only between CONTINUOUS frames.  A run of simple spans is modelled as one
    # frame per bent, where the bent's own tributary already IS the frame mass
    # -- the half-span each side is counted once, in the bent it belongs to.
    # Sharing a joint bent into such a frame would count the span beyond the
    # joint a second time, which is not what an expansion joint does.
    declared = {f.key: len(f.members) for f in frames}
    for i, f in enumerate(frames):
        if declared[f.key] < 2:
            continue
        edges = ((f.members[0], frames[i - 1] if i else None),
                 (f.members[-1], frames[i + 1] if i + 1 < len(frames) else None))
        for edge, nb in edges:
            if nb is None or edge.deck_link != "bearing":
                continue
            if declared[nb.key] < 2:
                continue
            if any(m.name == edge.name for m in nb.members):
                continue
            nb.members.append(edge)
            nb.members.sort(key=lambda b: b.order)
            nb.mass_share[edge.name] = 0.5
            f.mass_share[edge.name] = 0.5

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
        for key in b.frames:
            groups.setdefault(key, []).append(b)
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

        # --- stiffness across a SIMPLE SPAN, only if asked for ---
        # Two consecutive piers that are each alone in their frame are the two
        # supports of one simply supported span.  The span is a deck segment
        # between joints -- a frame -- so its supports can be held to the
        # adjacent-bent limit.  Whether they should be is a modelling choice,
        # hence the switch.
        if criteria.simple_span_stiffness:
            # DECLARED by one bent -- a continuous frame that happens to have
            # a single member in this direction (its others being released
            # here) is not a simple span and must not be caught.
            solo = [f for f in frames if f.simple_span and len(f.members) == 1]
            for fi, fj in zip(solo, solo[1:]):
                bi, bj = fi.members[0], fj.members[0]
                for bound in range(min(fi.n_bounds, fj.n_bounds)):
                    ki = bi.kappa(direction, bound, criteria.mass_normalized)
                    kj = bj.kappa(direction, bound, criteria.mass_normalized)
                    r = _ratio(ki, kj)
                    bad = math.isnan(r)
                    checks.append(BalanceCheck(
                        name=STIFFNESS_CHECK, pair=(bi.name, bj.name),
                        bound=fi.label(bound), ratio=r,
                        limit=criteria.k_ratio_min,
                        passed=(not bad) and r >= criteria.k_ratio_min,
                        direction=direction,
                        scope=f"{fi.key}–{fj.key}",
                        ref=criteria.ref_stiffness,
                        note=("stiffness unavailable (unstable p-y solve?)"
                              if bad else
                              f"simply supported span: "
                              f"{criteria.kappa_symbol} = {ki:.4g} vs {kj:.4g}"),
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
                      multiplier: float, silo: float,
                      end_fixity: str = "free") -> float:
    """Elastic lateral stiffness of the two-segment member at silo ``silo``.

    Reuses :meth:`Geometry.lateral_stiffness` with the silo swapped in, holding
    ``EI`` fixed.  Monotonically decreasing in ``silo`` under either end
    condition, which is what lets :func:`required_silo` bisect on it.
    """
    return replace(geometry, silo=silo).lateral_stiffness(
        EI_col, EI_shaft, multiplier, end_fixity=end_fixity)


def required_silo(geometry: Geometry, EI_col: float, EI_shaft: float,
                  multiplier: float, k_target: float,
                  silo_min: float = 0.0, silo_max: float = 240.0,
                  tol: float = 1e-3, end_fixity: str = "free") -> float | None:
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
            geometry, EI_col, EI_shaft, multiplier, silo_min,
            end_fixity) <= k_target else None
    if stiffness_at_silo(geometry, EI_col, EI_shaft, multiplier, silo_min,
                         end_fixity) <= k_target:
        return silo_min
    if stiffness_at_silo(geometry, EI_col, EI_shaft, multiplier, silo_max,
                         end_fixity) > k_target:
        return None                                   # unreachable within the cap
    lo, hi = silo_min, silo_max                       # k(lo) > target >= k(hi)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if stiffness_at_silo(geometry, EI_col, EI_shaft, multiplier, mid,
                             end_fixity) > k_target:
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
            groups.setdefault(b.frames[0] if b.frames else "", []).append(b)
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
